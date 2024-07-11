import os
import cv2
def bitwise_and_images(input_folder1, input_folder2, output_folder):
    # 确保输出文件夹存在，如果不存在，则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 获取输入文件夹1中的图像文件列表
    image_files1 = os.listdir(input_folder1)

    for image_file1 in image_files1:
        # 构建输入图像1和图像2的路径
        image_path1 = os.path.join(input_folder1, image_file1)
        image_path2 = os.path.join(input_folder2, image_file1)

        # 检查图像2是否存在
        if not os.path.exists(image_path2):
            continue

        # 加载图像1和图像2
        image1 = cv2.imread(image_path1, cv2.IMREAD_COLOR)
        image2 = cv2.imread(image_path2, cv2.IMREAD_COLOR)

        if image1 is not None and image2 is not None:
            # 执行并运算
            bitwise_and = cv2.bitwise_and(image1, image2)

            # 构建输出图像路径
            output_path = os.path.join(output_folder, image_file1)

            # 保存结果图像
            cv2.imwrite(output_path, bitwise_and)


# 示例用法
input_folder1 = '/media/zkk/duck/VL-PLM/datasets/coco/test/ori_result/'
input_folder2 = '/media/zkk/duck/VL-PLM/datasets/coco/test/PL_labels/'
output_folder = '/media/zkk/duck/VL-PLM/datasets/coco/test/ronghe/'
bitwise_and_images(input_folder1, input_folder2, output_folder)
