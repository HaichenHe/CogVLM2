import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM
import cv2
import json

# 设置为离线模式
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# 模型和设备配置
MODEL_PATH = "/opt/data/private/hhc/workdir/CogVLM2/model"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# 加载模型和分词器
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=TORCH_TYPE,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).eval()
print("Model loaded successfully!")

# 视频列表
video_paths = [
    "/opt/data/private/hhc/workdir/CogVLM2/haichen_demo/video/ballet.mp4",  # 替换为第一个视频路径
    "/opt/data/private/hhc/workdir/CogVLM2/haichen_demo/video/ice_fishing.mp4"  # 替换为第二个视频路径
]
T = 8  # 每个视频抽取的帧数

# 输出 JSON 文件路径
output_json_path = "captions.json"

# 如果文件不存在，创建空的 JSON 文件
if not os.path.exists(output_json_path):
    with open(output_json_path, "w") as f:
        json.dump({}, f, indent=4)

# 循环处理每个视频
for video_path in video_paths:
    print(f"Processing video: {video_path}")
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(total_frames // T, 1)  # 根据视频总帧数和T计算帧间隔

    captions = []  # 用于存储描述的列表

    print(f"Total frames in video: {total_frames}, FPS: {fps}, Extracting {T} frames...")

    frame_count = 0
    selected_frames = 0

    while cap.isOpened() and selected_frames < T:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:  # 按间隔抽取帧
            # 将帧从 BGR 转为 RGB，并转为 PIL 格式
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            # 构造输入
            query = "In one sentence, describe the detail and movement of objects, and the action and pose of persons in the image. Prioritize accuracy."
            history = []
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=history,
                images=[image],
                template_version='chat'
            )

            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
                'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]]
            }

            gen_kwargs = {
                "max_new_tokens": 1024,
                "pad_token_id": 128002,
            }

            # 使用模型生成描述
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0])
                response = response.split("<|end_of_text|>")[0]
                print(f"Caption for frame {frame_count}: {response}")

            # 将结果存入列表
            captions.append(response)
            selected_frames += 1

        frame_count += 1

    # 释放视频对象
    cap.release()

    # 将当前视频的 `captions` 写入 JSON 文件
    with open(output_json_path, "r") as f:
        data = json.load(f)

    # 使用视频文件名作为键存储（不使用 frame_xxx）
    video_name = os.path.basename(video_path)
    data[video_name] = captions

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Captions for video '{video_name}' saved to {output_json_path}")

print("All videos processed successfully!")
