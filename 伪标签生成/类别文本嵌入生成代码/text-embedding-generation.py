



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
import torchvision
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


# from nltk.corpus import wordnet
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


print(clip.available_models())
# with open('data/coco/coco_gzsd_2014_16_4.yaml') as f:
with open('data/VOC2007/voc_zsd_2007_val_16_4.yaml') as f:
    meta = yaml.safe_load(f)

    unseen_names = [meta['all_names'][i] for i in meta['val_names']]
    seen_names = [meta['all_names'][i] for i in meta['train_names']]
    # unseen_names = [meta['all_names'][i] for i in meta['unseen_names']]
    # seen_names = [meta['all_names'][i] for i in meta['seen_names']]

    all_names = meta['all_names']

    print(unseen_names)
    print(seen_names)
    print(all_names)

    defs = {i: wordnet.synsets(i)[0].definition() if len(wordnet.synsets(i)) else '' for i in all_names}
#
# unseen_names[10] = 'hotdog'
# unseen_names[12] = 'computer mouse'
defs_and_all_names = [i + ', ' + defs[i] + ',' if defs.get(i) else i for i in all_names]
templates = ['a photo of {} in the scene']
model, preprocess = clip.load('ViT-B/32')
def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname) for template in templates] #format with class
            texts = clip.tokenize(texts).cuda() #tokenize
            class_embeddings = model.encode_text(texts) #embed with text encoder
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights



seen_text_embeddings = zeroshot_classifier(seen_names, templates)
unseen_text_embeddings = zeroshot_classifier(unseen_names, templates)
all_text_embeddings = zeroshot_classifier(defs_and_all_names, templates)
#all_text_embeddings = zeroshot_classifier(meta['all_names'], templates)
torch.save(unseen_text_embeddings.T, 'embeddings/unseen_voc_text_embeddings_16_4.pt')
torch.save(seen_text_embeddings.T, 'embeddings/seen_voc_text_embeddings_16_4.pt')
# torch.save(all_text_embeddings.T, 'embeddings/all_voc_text_embeddings_16_4.pt')