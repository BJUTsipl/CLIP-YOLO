import json

# 指定要剔除的类别ID列表
# excluded_categories = [6,11,17,18]
excluded_categories = [0,1,2,3,4,
                       5,7,8,9,10,
                       12,13,14,15,16,
                       19]
#excluded_categories = []
# 读取原始的标注文件
original_json_file = 'output/annotations/val.json'
with open(original_json_file, 'r') as f:
    data = json.load(f)

# 进行类别的重新映射
category_id_map = {}
new_categories = []
new_annotations = []

for category in data['categories']:
    if category['id'] not in excluded_categories:
       # new_id = len(new_categories) + 1 # 重新映射的ID从1开始
        new_id = len(new_categories)  # 重新映射的ID从1开始
        category_id_map[category['id']] = new_id
        category['id'] = new_id
        new_categories.append(category)

for annotation in data['annotations']:
    if annotation['category_id'] not in excluded_categories:
        annotation['category_id'] = category_id_map[annotation['category_id']]
        new_annotations.append(annotation)

data['categories'] = new_categories
data['annotations'] = new_annotations

# 保存重新映射后的标注文件
new_json_file = './val_3090_ext.json'
with open(new_json_file, 'w') as f:
    json.dump(data, f)