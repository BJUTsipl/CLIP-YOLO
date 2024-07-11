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


with open('data/hyp.scratch.yaml') as f:
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

    #创建输出文件夹
    if os.path.exists(out_path) and delete:
        shutil.rmtree(out_path)
        os.mkdir(out_path)

    clip_model, preprocess = clip.load(clip_name)
    #这里加载yolo模型
    model = load_model(model_path, hyp).eval() if model_path else None
    #步长设置
    gs = max(int(model.stride.max()), 32) if model else 32
    #这个用不到
    loader, _ = (loader, None) if loader else create_dataloader(path, imgsz, batch_size, gs, opt, hyp=hyp, workers=4)
    #clip预处理参数设置
    preprocess.transforms = [Resize(size=(224, 224)), lambda x: x.type(torch.cuda.HalfTensor),
                             preprocess.transforms[-1]]
    #变量初始化
    removed_boxes, total_boxes, self_label_boxes = 0, 0, 0
    #进度条
    pbar = tqdm(loader, total=len(loader))
    for data in pbar:
        #获取图片数量
        c_batch_size = len(data[0])
        #创建一个数组用来统计每张图片中bbox的数量
        count_per_batch = [0, ] * c_batch_size
        #data【1】是注释，z中的注释
        for i in data[1]:

            count_per_batch[int(i[0])] += 1  # [4,1,3,0,0...0]
        '''
        根据每个图像中的框数将标注数据 data[1] 分割成多个子列表，
        每个子列表包含一个图像中的所有标注框。
        '''
        split_boxes = data[1].split(count_per_batch)
        '''
        对每个子列表中的标注框进行坐标转换，
        将标注框的格式从 (x, y, width, height) 转换为 (x1, y1, x2, y2)。
        '''
        for i in range(len(split_boxes)):
            split_boxes[i][:, 2:] = xywhn2xyxy(split_boxes[i][:, 2:])
        '''
          将每个子列表中的标注框进行处理和拼接，创建一个新的列表 split_boxes。
          处理包括将标注框的坐标转换为相对于图像的归一化坐标，
          添加一个置信度值，以及将结果转移到 GPU 上进行加速。
          '''
        split_boxes = [
            torch.cat([i[..., 2:], torch.ones((i.shape[:-1] + (1,))) + 0.1, i[..., 1].unsqueeze(-1)], dim=1).cuda() for
            i in split_boxes]
        # if model:
        #     #将图像数据 data[0] 转移到 GPU 上，并进行归一化处理。
        #     imgs = data[0].to('cuda', non_blocking=True).float() / 255
        #     with torch.no_grad():
        #         #：使用模型对图像数据进行推断，生成预测结果 output。由于使用了 torch.no_grad()，在此过程中不会进行梯度计算。
        #
        #         output = model(imgs)
        #         '''
        #                        将预测结果 output 中的边界框与之前处理的标注框 `split_boxes
        #                        进行拼接，创建一个新的列表 all_boxes。
        #                        拼接的结果包括预测边界框的类别、置信度和坐标信息，以及之前处理的标注框。
        #                        (包含便签和预测结果)
        #                        '''
        #     for idx, i in enumerate(output[0]):
        #         i[:, :4] = xywh2xyxy(i[:, :4])
        #         i[:, 5] = -1
        #     all_boxes = [torch.cat([output[0][i][:, :6], split_boxes[i]]) for i in range(len(split_boxes))]
        #     '''
        #              根据置信度阈值 score_thresh 对每个图像的边界框进行筛选，
        #              只保留置信度高于阈值的边界框。
        #              '''
        #     all_boxes = [i[i[:, 4] > score_thresh] for i in all_boxes]
        #     '''
        #               对每个图像的边界框应用非最大抑制 (NMS)，去除重叠度高的边界框，只保留最具代表性的边界框。
        #               '''
        #     all_boxes = [i[nms(i[:, :4], i[:, 4], iou_threshold=iou_thresh)] for i in all_boxes]
        #     # 根据类别索引，去除类别为背景的边界框。
        #     all_boxes = [i[i[:, 4] < 1] for i in all_boxes]
        #     # 在每个边界框末尾添加一个标记，表示该边界框不再需要进行处理。
        #     all_boxes = [torch.cat([i[..., :-1], torch.zeros(i[...].shape[:-1] + (1,)).cuda() - 1], dim=-1) for i in
        #                  all_boxes]
        #     for i in range(len(all_boxes)):
        #         # 对每个图像的边界框计算宽度和高度是否满足最小尺寸要求，生成一个布尔掩码。
        #         mask = torch.Tensor(
        #             [(int(j[2]) - int(j[0])) > min_w and (int(j[3]) - int(j[1])) > min_h for j in all_boxes[i]])
        #         #根据布尔掩码对边界框进行筛选，只保留宽度和高度满足最小尺寸要求的边界框。
        #
        #         all_boxes[i] = all_boxes[i][mask.type(torch.BoolTensor)]
        #         #将之前处理的标注框 split_boxes 与筛选后的边界框 all_boxes 进行拼接，更新 split_boxes 列表。
        #
        #     split_boxes = [torch.cat([split_boxes[i], all_boxes[i]]) for i in range(len(split_boxes))]
        for i in range(len(split_boxes)):
            # 对每个图像的边界框的 x 坐标进行裁剪，确保边界框在图像范围内。

            split_boxes[i][:, 0] = torch.clip(split_boxes[i][:, 0], min=data[3][i][1][1][0],
                                              max=data[3][i][0][1] * data[3][i][1][0][1] + data[3][i][1][1][0])
            # 对每个图像的边界框的 y 坐标进行裁剪，确保边界框在图像范围内。

            split_boxes[i][:, 1] = torch.clip(split_boxes[i][:, 1], min=data[3][i][1][1][1],
                                              max=data[3][i][0][0] * data[3][i][1][0][0] + data[3][i][1][1][1])
            # 对每个图像的边界框的高度进行裁剪，确保边界框在图像范围内。

            split_boxes[i][:, 2] = torch.clip(split_boxes[i][:, 2], min=data[3][i][1][1][0],
                                              max=data[3][i][0][1] * data[3][i][1][0][1] + data[3][i][1][1][0])
            # 对每个图像的边界框的宽度进行裁剪，确保边界框在图像范围内。

            split_boxes[i][:, 3] = torch.clip(split_boxes[i][:, 3], min=data[3][i][1][1][1],
                                              max=data[3][i][0][0] * data[3][i][1][0][0] + data[3][i][1][1][1])
            if remove_tiny:
                #如果设置了 remove_tiny 标志为 True，则计算每个边界框的宽度和高度是否大于1，生成一个布尔掩码。

                mask = torch.Tensor([(((j[2] - j[0]) > 1) and ((j[3] - j[1]) > 1)) for j in split_boxes[i]])
                # 根据布尔掩码对边界框进行筛选，去除宽度或高度小于等于1的边界框，并统计被移除的边界框数量。

                previous_len = len(split_boxes[i])
                split_boxes[i] = split_boxes[i][mask.type(torch.BoolTensor)]
                removed_boxes += previous_len - len(split_boxes[i])
                # 根据测试标志 test，如果为 True，则创建与边界框数量相同的空张量列表 embeddings；否则，根据边界框的坐标提取图像嵌入向量。

        embeddings = [torch.zeros((i.shape[0], 512)) for i in split_boxes] if test else extract_image_embeddings(
            data[0], [i[:, :4] for i in split_boxes], clip_model, preprocess)
        # print([i.shape for i in embeddings])
        for i in range(len(split_boxes)):
            #将每个图像的边界框的坐标格式从 (x1, y1, x2, y2) 转换回 (x, y, width, height)，并进行归一化处理。
            split_boxes[i][:, :4] = xyxy2xywhn(split_boxes[i][:, :4], w=data[3][i][0][1] * data[3][i][1][0][1],
                                               h=data[3][i][0][0] * data[3][i][1][0][0], padw=data[3][i][1][1][0],
                                            padh=data[3][i][1][1][1])

        '''
        创建一个注释列表 annot，其中每个注释包括边界框的类别索引、坐标信息和对应的图像嵌入向量。
        这里使用 torch.cat 进行张量的拼接，并将结果移动到 CPU 上。
        '''

        annot = [torch.cat([split_boxes[i][:, 5].unsqueeze(-1), split_boxes[i][:, :4], embeddings[i]], dim=1).cpu() for
                 i in range(len(split_boxes))]
        '''
                将注释列表 annot 保存为 Torch 文件，保存路径为 out_path。
                '''
        #  for i in annot:
        #   '''
        #     统计注释中自标注的边界框数量，即类别索引为 10000 的边界框。

        #    '''
            # if sum(i[:, 0] == 10000):
            #   i[:, 0] = -1

        for i in annot:
            # 将类别标签为10000的边界框的类别标签更改为-1
            i[i[:, 0] == 10000, 0] = -1

        save_annot_torch(annot, data, out_path)
        total_boxes += sum(len(i) for i in split_boxes)
        # for i in annot:
        #     '''
        #              统计注释中自标注的边界框数量，即类别索引为 -1 的边界框。
        #              '''
        #    self_label_boxes += sum(i[:, 0] == -1)
        for i in annot:
            '''
                     统计注释中自标注的边界框数量，即类别索引为 10000 的边界框。
                     
                     '''


            #self_label_boxes += sum(i[:, 0] == -1)
            self_label_boxes += sum(i[:, 0] == -1)
        pbar.desc = f'Total removed boxes: {removed_boxes}. Total generated boxes: {total_boxes}. Self-label boxes: {self_label_boxes}. Generating Embeddings to {out_path}.1'
# loader, _ = create_dataloader('data/coco/images2014_zsd_test_48_17/val2014',
#                               640, 16, 32, opt, hyp=hyp, workers=8, raw=False,
#                               annot_folder='48_17_label_txt')
# generate_zsd_data('data/coco/images2014_zsd_test_48_17/val2014', hyp, opt,
#                                       '/media/ubuntu/dataset/ZSD-YOLO/data/coco/label_48_17_10_17/val2014', model_path='weights/pretrained_weights/yolov5l.pt',
#                                       loader=loader, min_w=25, min_h=25, iou_thresh=0.2, score_thresh=0.3, delete=False, test=False)
# loader, _ = create_dataloader('data/coco/images2014_zsd_test_48_17/train2014',
#                               640, 16, 32, opt, hyp=hyp, workers=8, raw=False,
#                               annot_folder='labels_pl_48_17_10000')
# generate_zsd_data('data/coco/images2014_zsd_test_48_17/train2014', hyp, opt,
#                                       'data/coco/48_17_VL_pt/train2014', model_path='weights/pretrained_weights/yolov5l.pt',
#                                       loader=loader, min_w=25, min_h=25, iou_thresh=0.2, score_thresh=0.3, delete=False, test=False)
# loader, _ = create_dataloader('data/voc2007_2012/images/val',
#                               640, 16, 32, opt, hyp=hyp, workers=8, raw=False,
#                               annot_folder='labels') #LABEL .TXT
# generate_zsd_data('data/voc2007_2012/images/val', hyp, opt,
#                                       '/media/ubuntu/dataset/ZSD-YOLO/data/voc2007_2012/val_pt', model_path='weights/pretrained_weights/yolov7x.pt',
#                                       loader=loader, min_w=25, min_h=25, iou_thresh=0.2, score_thresh=0.3, delete=False, test=False)


# loader, _ = create_dataloader('/media/ubuntu/dataset/ZSD-YOLO/data/coco/images2014_zsd_test_48_17/train2014',
#                               640, 16, 32, opt, hyp=hyp, workers=8, raw=False,
#                               annot_folder='labels_pl_48_17_10000') #LABEL .TXT  #annot_folder='labels_no_del'
# generate_zsd_data('/media/ubuntu/dataset/ZSD-YOLO/data/coco/images2014_zsd_test_48_17', hyp, opt,
#                                       '/media/ubuntu/dataset/ZSD-YOLO/data/coco/label_train_zk/train2014', model_path='weights/pretrained_weights/yolov7x.pt',
#                                       loader=loader, min_w=25, min_h=25, iou_thresh=0.2, score_thresh=0.3, delete=False, test=False)
#
#
loader, _ = create_dataloader('/media/ubuntu/dataset/ZSD-YOLO/data/coco/images2014_zsd_val_65_15/train2014',
                              640, 16, 32, opt, hyp=hyp, workers=8, raw=False,
                              annot_folder='label_pl_zk') #LABEL .TXT  #annot_folder='labels_no_del'
generate_zsd_data('/media/ubuntu/dataset/ZSD-YOLO/data/coco/images2014_zsd_test_65_15/train2014', hyp, opt,
                                      '/media/ubuntu/dataset/ZSD-YOLO/data/coco/label_train_zk/train2014', model_path='weights/pretrained_weights/yolov7x.pt',
                                      loader=loader, min_w=25, min_h=25, iou_thresh=0.2, score_thresh=0.3, delete=False, test=False)


