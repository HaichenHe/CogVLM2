import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import csv
import slowfast.utils.distributed as du
from slowfast.datasets import loader
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from torchvision.transforms import ToPILImage, ToTensor
import matplotlib.pyplot as plt


def denormalize(tensor, mean, std):
    mean = torch.tensor(mean, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(-1, 1, 1)
    return tensor * std + mean


def process_video(video_data, model, tokenizer, index, output_file):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    frame_descriptions = []  
    c, t, h, w = video_data.shape

    for frame_idx in range(t):
        frame = video_data[:, frame_idx, :, :]  # Shape: (c, h, w)

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        frame = denormalize(frame, mean, std)

        frame = torch.clamp(frame, 0, 1)
        frame_pil = ToPILImage()(frame)

        # plt.subplot(1, 2, 1)
        # plt.imshow(frame.permute(1, 2, 0).cpu().numpy())  
        # plt.title("Original Tensor")

        # plt.subplot(1, 2, 2)
        # plt.imshow(frame_pil)
        # plt.title("PIL Image")

        # plt.savefig(f"/opt/data/private/hhc/workdir/CogVLM2/generate_prompts/visualization_{video_index}_frame_{frame_idx}.png")
        # plt.close()


        query = "In one sentence, describe the detail and movement of objects, and the action and pose of persons in the image. Prioritize accuracy."
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=query,
            history=[],
            images=[frame_pil],
            template_version='chat'
        )

        inputs = {
            'input_ids': input_by_model['input_ids'].unsqueeze(0).to(DEVICE),
            'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(DEVICE),
            'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(DEVICE),
            'images': [[input_by_model['images'][0].to(DEVICE).to(TORCH_TYPE)]],
        }

        gen_kwargs = {
            "top_k": 1,
            "top_p": 0.6,
            "temperature": 0.8,
            "max_new_tokens": 1024,
            "pad_token_id": 128002,
        }

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(outputs[0])
            response = response.split("<|end_of_text|>")[0]

        frame_descriptions.append(response)

    try:
        with open(output_file, "r", encoding="utf-8") as f:
            all_video_descriptions = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        all_video_descriptions = {} 

    all_video_descriptions[index] = frame_descriptions

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_video_descriptions, f, indent=4, ensure_ascii=False)

    print(f"Processed and saved video {index} descriptions to {output_file}")


def get_video_index(cfg, path_to_video, labels):
    train_full = cfg.TRAIN_FULL_FILE

    train_video_full = []
    with open(train_full, 'r', encoding='utf-8') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            train_video_full.append(row)
    

    path_to_video = path_to_video[0]
    labels = int(labels[0])
    relative_path_to_video = path_to_video.split("/train/")[-1]
    relative_path_to_video = "train/" + relative_path_to_video
    row = [relative_path_to_video, str(labels)] 
    # row = f"{relative_path_to_video},{labels}" # train/playing_drums/TUlpPhhUbX8_000000_000010.mp4,230

    try:
        return train_video_full.index(row)
    except:
        return -1



def perform_generate(cfg, train_loader, model, tokenizer):
    output_file = cfg.OUTPUT_GENERATE_PROMPT

    for cur_iter, (inputs, labels, index, time, meta, path_to_video) in enumerate(train_loader):
        # Transfer the data to the current GPU device.
        if cfg.NUM_GPUS:
            if isinstance(inputs, (list,)):
                for i in range(len(inputs)):
                    if isinstance(inputs[i], (list,)):
                        for j in range(len(inputs[i])):
                            inputs[i][j] = inputs[i][j].cuda(non_blocking=True)
                    else:
                        inputs[i] = inputs[i].cuda(non_blocking=True)
            else:
                inputs = inputs.cuda(non_blocking=True)
            if not isinstance(labels, list):
                labels = labels.cuda(non_blocking=True)
                index = index.cuda(non_blocking=True)
                time = time.cuda(non_blocking=True)
            for key, val in meta.items():
                if isinstance(val, (list,)):
                    for i in range(len(val)):
                        val[i] = val[i].cuda(non_blocking=True)
                else:
                    meta[key] = val.cuda(non_blocking=True)

        
        index = get_video_index(cfg, path_to_video, labels)
        
        # image processing
        inputs = inputs[0]
        bz, channel_dim, clip_len, h, w = inputs.shape

        for video_idx in range(bz):
            process_video(inputs[video_idx], model, tokenizer, index, output_file)


def generate_video_specific_prompt(cfg):
    # Set up environment.
    try:
        du.init_distributed_training(cfg)
    except:
        du.init_distributed_training(cfg.NUM_GPUS, cfg.SHARD_ID)

    # Set random seed from configs.
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    MODEL_PATH = "/opt/data/private/hhc/workdir/CogVLM2/model"
    TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=TORCH_TYPE, trust_remote_code=True, low_cpu_mem_usage=True).eval()
    print("Model loaded successfully!")

    # Create the video train and val loaders.
    train_loader = loader.construct_loader(cfg, "train")

    perform_generate(cfg, train_loader, model, tokenizer)
