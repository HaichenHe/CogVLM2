from collections import OrderedDict

# 文件路径
file_path = '/opt/data/private/hhc/workdir/CogVLM2/generate_prompts/split_18/train.csv'

# 用于存储类别的有序集合
categories = OrderedDict()

# 打开文件并解析内容
with open(file_path, 'r') as file:
    for line in file:
        # 提取第一个和第二个斜杠之间的字符串
        parts = line.strip().split('/')
        if len(parts) > 2:
            category = parts[1]
            # 按出现顺序添加到有序集合
            if category not in categories:
                categories[category] = True

# 将类别按出现顺序提取
ordered_categories = list(categories.keys())

# 按行输出类别
print(f"共有 {len(ordered_categories)} 种类别：")
for category in ordered_categories:
    print(category)