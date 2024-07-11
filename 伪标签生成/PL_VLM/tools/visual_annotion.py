import os
import random

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

# COCO数据集标注文件的路径
#annotations_path = '/media/zkk/duck/VL-PLM/datasets/coco/annotations/val.json'
#annotations_path = '/media/zkk/duck/VL-PLM/tools/railway_base_novel.json'
#annotations_path = '/media/zkk/duck/VL-PLM/tools/railway_novel.json'
#annotations_path = '/media/zkk/duck/VL-PLM/datasets/coco/annotations/val.json'
annotations_path = '/media/zkk/duck/VL-PLM/tools/test_PL.json'
# COCO数据集图像文件夹的路径
image_folder_path = '/media/zkk/duck/VL-PLM/datasets/coco/test/ori_result'

# 保存可视化结果的文件夹路径
output_folder_path = '/media/zkk/duck/VL-PLM/datasets/coco/test/ronghe1'

# 初始化COCO对象
coco = COCO(annotations_path)

# 获取验证集所有图像的ID
image_ids = coco.getImgIds()

# 遍历每个图像并可视化标注框和类别
for image_id in image_ids:
    # 加载图像
    image_info = coco.loadImgs(image_id)[0]
    image_path = os.path.join(image_folder_path, image_info['file_name'])
    image = cv2.imread(image_path)

    # 获取该图像的所有标注框
    annotation_ids = coco.getAnnIds(imgIds=image_id)
    annotations = coco.loadAnns(annotation_ids)

    # 在图像上绘制标注框和类别
    for annotation in annotations:
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        category_info = coco.loadCats(category_id)[0]
        category_name = category_info['name']

        x, y, w, h = bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(image, (x, y), (x + w, y + h), (random.randint(1,255),random.randint(1,255), random.randint(1,255)), 2)
        cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 保存可视化结果
    output_path = os.path.join(output_folder_path, image_info['file_name'])
    cv2.imwrite(output_path, image)
