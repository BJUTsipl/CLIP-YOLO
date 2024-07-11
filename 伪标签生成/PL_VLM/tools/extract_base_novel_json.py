import os
import json
from pycocotools.coco import COCO

def extract_coco_info(images_folder, coco_json, output_json):
    # 加载COCO JSON文件
    coco = COCO(coco_json)

    # 获取文件夹中的图像文件列表
    image_files = os.listdir(images_folder)

    # 创建存储提取信息的字典
    output_data = {'images': [], 'annotations': []}

    # 遍历图像文件列表，提取与每个图像对应的信息，并将其添加到字典中
    for image_file in image_files:
        # 构建图像文件的完整路径
        image_path = os.path.join(images_folder, image_file)

        # 通过图像文件名获取图像ID
        image_id = coco.getImgIds(imgIds=[image_file])[0]
        print(image_id)
        # 获取与图像ID对应的注释信息
        annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id]))
        print(annotations)
        # 处理注释信息
        for annotation in annotations:
            # 将注释信息添加到字典的annotations列表中
            output_data['annotations'].append(annotation)

        # 处理图像信息
        image_info = coco.loadImgs(image_id)[0]

        # 将图像信息添加到字典的images列表中
        output_data['images'].append(image_info)

    # 将提取的信息保存为JSON文件
    with open(output_json, 'w') as f:
        json.dump(output_data, f)

# 示例用法
images_folder = '/media/zkk/duck/VL-PLM/datasets/coco/PL_results/'
coco_json = '/media/zkk/duck/VL-PLM/datasets/coco/annotations/train.json'
output_json = 'base_novel.json'
extract_coco_info(images_folder, coco_json, output_json)
