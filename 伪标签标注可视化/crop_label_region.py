import os
import cv2
import json
from pycocotools.coco import COCO

def crop_annotations(coco_json, images_folder, output_folder):
    # 加载COCO JSON文件
    coco = COCO(coco_json)

    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取所有类别
    categories = coco.loadCats(coco.getCatIds())

    # 创建以类别命名的文件夹
    for category in categories:
        category_folder = os.path.join(output_folder, category['name'])
        if not os.path.exists(category_folder):
            os.makedirs(category_folder)

    # 获取所有图像ID
    image_ids = coco.getImgIds()

    # 遍历图像ID
    for image_id in image_ids:
        # 加载图像信息
        image_info = coco.loadImgs(image_id)[0]

        # 构建图像文件路径
        image_path = os.path.join(images_folder, image_info['file_name'])

        # 加载图像
        image = cv2.imread(image_path)

        # 获取与图像ID对应的注释信息
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))

        # 遍历注释信息
        for annotation in annotations:
            # 获取标注区域的边界框坐标
            bbox = annotation['bbox']
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # 裁剪标注区域
            cropped_region = image[y:y+h, x:x+w]

            # 获取标注类别名称
            category_id = annotation['category_id']
            category_info = coco.loadCats(category_id)[0]
            category_name = category_info['name']

            # 构建保存路径
            save_path = os.path.join(output_folder, category_name, f'{image_id}_{category_id}.jpg')

            # 保存裁剪后的图像
            cv2.imwrite(save_path, cropped_region)

# 示例用法
coco_json = '/media/zkk/duck/VL-PLM/tools/railway_novel.json'
images_folder = '/media/zkk/duck/VL-PLM/datasets/coco/images/'
output_folder = '/media/zkk/duck/VL-PLM/datasets/coco/test/crop_PL_region/'
crop_annotations(coco_json, images_folder, output_folder)