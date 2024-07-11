#coding=UTF-8

'''

通过命令行参数传递原始COCO JSON文件的路径和要保存的新JSON文件的路径。

定义了binary_search_loop函数，用于在列表中执行二分查找。

定义了要使用的新颖类别的名称，可以在代码中修改为所需的类别集合。

使用pycocotools库的COCO类加载原始JSON文件。

获取所需新颖类别的类别ID。

使用json模块加载原始JSON文件的内容。

创建新的注释列表和图像ID列表。

遍历原始注释列表，并根据类别ID筛选出所需的新颖类别的注释。将这些注释添加到新的注释列表，并记录相应的图像ID。

对图像ID列表进行排序，并在原始图像信息列表中找到对应的图像信息，并将其添加到新的图像信息列表。

输出新的注释数量、图像数量以及每个图像的平均注释数。

将更新后的注释列表和图像信息列表写入新的JSON文件中。

该代码主要用于从COCO数据集中选择特定类别的注释，并将其保存到一个新的JSON文件中。这对于开放词汇检测任务中需要使用特定类别的注释很有用。





'''

import json
import os, glob, argparse

import numpy as np

from pycocotools.coco import COCO

from utils import COCO_BASE_CatName as baseCatNames
from utils import COCO_NOVEL_CatName as novelCatNames


def binary_search_loop(lst, value):
    low, high = 0, len(lst) - 1
    while low <= high:
        mid = (low + high) // 2
        if lst[mid] < value:
            low = mid + 1
        elif lst[mid] > value:
            high = mid - 1
        else:
            return mid
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect novel category annotations from COCO json file for open vocabulary detection')
    parser.add_argument('coco_json',type=str,default='../datasets/coco/annotations/train.json', help='original coco json file to split')
    parser.add_argument('save_json',type=str,default='./train_novel.json', help='split json file')

    args = parser.parse_args()

    ################################
    orig_json_file = args.coco_json         # '../datasets/coco/annotations/instances_val2017.json'
    save_json_file = args.save_json         # './inst_val2017_novel.json'

    usedCatNames = novelCatNames            # change to required category set if needed

    ################################

    coco = COCO(orig_json_file)
    #print('------------coco--------------------')
  #  print(coco)
    print('----------usedCatIds----------------------')
    usedCatIds = coco.getCatIds(catNms=usedCatNames)
    print(usedCatIds)
    data = json.load(open(orig_json_file, 'r'))

    ## annotations
    new_annotations_list = list()
    new_image_ids = list()

    for anno in data['annotations']:
        curImgId = anno['image_id']
        curCatId = anno['category_id']

        if curCatId in usedCatIds:
            new_annotations_list.append(anno)
            new_image_ids.append(curImgId)
            # print("---------new anootaion--------------------")
            # print(new_annotations_list)
            # print("---------new image id--------------------")
            # print(new_image_ids)
    new_image_ids = sorted(list(set(new_image_ids)))

    ## images
    new_imgInfo_list = list()
    for imgInfo in data['images']:
        curImgId = imgInfo['id']

        findCurId = binary_search_loop(new_image_ids, curImgId)
        if findCurId is not None:
            new_imgInfo_list.append(imgInfo)
            # print("---------new_imgInfo_list-------------")
            # print(new_imgInfo_list)

    # write to json file
    print( 'annotation num: %d, image num: %d, anno per image: %.2f'%(len(new_annotations_list), len(new_imgInfo_list), len(new_annotations_list)/len(new_imgInfo_list)))

    data['annotations'] = new_annotations_list
    data['images'] = new_imgInfo_list
    with open(save_json_file, 'w') as outfile:
        json.dump(data, outfile)

