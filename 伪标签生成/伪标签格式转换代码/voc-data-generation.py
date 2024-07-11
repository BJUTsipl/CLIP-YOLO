import argparse
import logging
import math
import os
import random
import shutil
import json
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread
from importlib import reload

import numpy as np
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import yaml
import clip
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.ops import nms
from tqdm import tqdm
from PIL import Image
from utils.general import xywhn2xyxy, xywh2xyxy, xyxy2xywh, xyxy2xywhn
from torchvision.transforms import Resize

import test  # import test.py to get mAP after each epoch
from nltk.corpus import wordnet
from models.experimental import attempt_load
from models.yolo import Model
from utils.general import non_max_suppression
from utils.autoanchor import check_anchors
from utils.datasets import create_dataloader, LoadZSD
from utils.general import labels_to_class_weights, increment_path, labels_to_image_weights, init_seeds, \
    fitness, strip_optimizer, get_latest_run, check_dataset, check_file, check_git_status, check_img_size, \
    check_requirements, print_mutation, set_logging, one_cycle, colorstr
from utils.google_utils import attempt_download
from utils.loss import ComputeLoss
from utils.plots import plot_images, plot_labels, plot_results, plot_evolution
from utils.torch_utils import ModelEMA, select_device, intersect_dicts, torch_distributed_zero_first, is_parallel
from utils.wandb_logging.wandb_utils import WandbLogger, check_wandb_resume

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
opt = Namespace(single_cls=False, zsd=False)

with open('/media/ubuntu/dataset/ZSD-YOLO/data/hyp.scratch.yaml') as f:
    hyp = yaml.safe_load(f)

def load_model(model_path, hyp, device='cuda'):
    ckpt = torch.load(model_path, map_location=device)
    model = Model(ckpt['model'].yaml, ch=3, anchors=hyp.get('anchors'), hyp=hyp).to(device)
    state_dict = ckpt['model'].float().state_dict()
    model.load_state_dict(state_dict, strict=False)
    print('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), model_path))
    return model
def extract_image_embeddings(images, boxes, clip_model, preprocess):
    all_embeddings = []
    for i in range(len(boxes)):
        bboxes = deepcopy(boxes[i]).type(torch.IntTensor)
        regions = []
        include = []
        for j in range(len(bboxes)):
            x1, y1, x2, y2 = [int(k) for k in bboxes[j]]
            regions.append(preprocess(images[i][:, y1:y2, x1:x2].clone().detach().float() / 255))
        if(len(regions)):
            regions = torch.stack(regions).cuda()
            with torch.no_grad():
                all_embeddings.append(clip_model.visual(regions))
        else:
            all_embeddings.append(torch.zeros((0, 512)).cuda())
    return all_embeddings
def save_annot_torch(annot, data, out_path):
    paths = [os.path.join(out_path, i.split('/')[-1].split('.')[0] + '.pt') for i in data[2]]
    for i in range(len(paths)):
        torch.save(annot[i].cpu(), paths[i])


def generate_zsd_data(path, hyp, opt, out_path, imgsz=640, batch_size=16, model_path=None, clip_name='ViT-B/32',
                      score_thresh=0.1, iou_thresh=0.1, loader=None, min_w=0, min_h=0, delete=False, test=False,
                      remove_tiny=True):
    if os.path.exists(out_path) and delete:
        shutil.rmtree(out_path)
        os.mkdir(out_path)

    clip_model, preprocess = clip.load(clip_name)
    model = load_model(model_path, hyp).eval() if model_path else None
    gs = max(int(model.stride.max()), 32) if model else 32
    loader, _ = (loader, None) if loader else create_dataloader(path, imgsz, batch_size, gs, opt, hyp=hyp, workers=4)
    preprocess.transforms = [Resize(size=(224, 224)), lambda x: x.type(torch.cuda.HalfTensor),
                             preprocess.transforms[-1]]
    removed_boxes, total_boxes, self_label_boxes = 0, 0, 0
    pbar = tqdm(loader, total=len(loader))
    for data in pbar:
        c_batch_size = len(data[0])
        count_per_batch = [0, ] * c_batch_size
        for i in data[1]:
            count_per_batch[int(i[0])] += 1
        split_boxes = data[1].split(count_per_batch)
        for i in range(len(split_boxes)):
            split_boxes[i][:, 2:] = xywhn2xyxy(split_boxes[i][:, 2:])
        split_boxes = [
            torch.cat([i[..., 2:], torch.ones((i.shape[:-1] + (1,))) + 0.1, i[..., 1].unsqueeze(-1)], dim=1).cuda() for
            i in split_boxes]
        if model:
            imgs = data[0].to('cuda', non_blocking=True).float() / 255
            with torch.no_grad():
                output = model(imgs)
            for idx, i in enumerate(output[0]):
                i[:, :4] = xywh2xyxy(i[:, :4])
                i[:, 5] = -1
            all_boxes = [torch.cat([output[0][i][:, :6], split_boxes[i]]) for i in range(len(split_boxes))]
            all_boxes = [i[i[:, 4] > score_thresh] for i in all_boxes]
            all_boxes = [i[nms(i[:, :4], i[:, 4], iou_threshold=iou_thresh)] for i in all_boxes]
            all_boxes = [i[i[:, 4] < 1] for i in all_boxes]
            all_boxes = [torch.cat([i[..., :-1], torch.zeros(i[...].shape[:-1] + (1,)).cuda() - 1], dim=-1) for i in
                         all_boxes]
            for i in range(len(all_boxes)):
                mask = torch.Tensor(
                    [(int(j[2]) - int(j[0])) > min_w and (int(j[3]) - int(j[1])) > min_h for j in all_boxes[i]])
                all_boxes[i] = all_boxes[i][mask.type(torch.BoolTensor)]
            split_boxes = [torch.cat([split_boxes[i], all_boxes[i]]) for i in range(len(split_boxes))]
        for i in range(len(split_boxes)):
            split_boxes[i][:, 0] = torch.clip(split_boxes[i][:, 0], min=data[3][i][1][1][0],
                                              max=data[3][i][0][1] * data[3][i][1][0][1] + data[3][i][1][1][0])
            split_boxes[i][:, 1] = torch.clip(split_boxes[i][:, 1], min=data[3][i][1][1][1],
                                              max=data[3][i][0][0] * data[3][i][1][0][0] + data[3][i][1][1][1])
            split_boxes[i][:, 2] = torch.clip(split_boxes[i][:, 2], min=data[3][i][1][1][0],
                                              max=data[3][i][0][1] * data[3][i][1][0][1] + data[3][i][1][1][0])
            split_boxes[i][:, 3] = torch.clip(split_boxes[i][:, 3], min=data[3][i][1][1][1],
                                              max=data[3][i][0][0] * data[3][i][1][0][0] + data[3][i][1][1][1])
            if remove_tiny:
                mask = torch.Tensor([(((j[2] - j[0]) > 1) and ((j[3] - j[1]) > 1)) for j in split_boxes[i]])
                previous_len = len(split_boxes[i])
                split_boxes[i] = split_boxes[i][mask.type(torch.BoolTensor)]
                removed_boxes += previous_len - len(split_boxes[i])
        embeddings = [torch.zeros((i.shape[0], 512)) for i in split_boxes] if test else extract_image_embeddings(
            data[0], [i[:, :4] for i in split_boxes], clip_model, preprocess)
        # print([i.shape for i in embeddings])
        for i in range(len(split_boxes)):
            split_boxes[i][:, :4] = xyxy2xywhn(split_boxes[i][:, :4], w=data[3][i][0][1] * data[3][i][1][0][1],
                                               h=data[3][i][0][0] * data[3][i][1][0][0], padw=data[3][i][1][1][0],
                                               padh=data[3][i][1][1][1])
        annot = [torch.cat([split_boxes[i][:, 5].unsqueeze(-1), split_boxes[i][:, :4], embeddings[i]], dim=1).cpu() for
                 i in range(len(split_boxes))]
        save_annot_torch(annot, data, out_path)
        total_boxes += sum(len(i) for i in split_boxes)
        for i in annot:
            self_label_boxes += sum(i[:, 0] == -1)
        pbar.desc = f'Total removed boxes: {removed_boxes}. Total generated boxes: {total_boxes}. Self-label boxes: {self_label_boxes}. Generating Embeddings to {out_path}.1'
# loader, _ = create_dataloader('/media/zkk/ZSD-YOLO/data/coco/images2014_zsd_val_65_15/train2014',
#                                   640, 16, 32, opt, hyp=hyp, workers=8, raw=False,
#                                   annot_folder='labels2014_zsd_split')
loader, _ = create_dataloader('data/voc2007_2012/images/train',
                              640, 16, 32, opt, hyp=hyp, workers=8, raw=False,
                              annot_folder='labels') #LABEL .TXT
generate_zsd_data('data/voc2007_2012/images/train', hyp, opt,
                                      '/media/ubuntu/dataset/ZSD-YOLO/data/voc2007_2012/train_pt_17', model_path='weights/pretrained_weights/yolov7x.pt',
                                      loader=loader, min_w=25, min_h=25, iou_thresh=0.2, score_thresh=0.3, delete=False, test=False)


