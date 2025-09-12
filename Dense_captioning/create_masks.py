import pandas as pd
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from datasets import load_dataset 
from transformers.pipelines.pt_utils import KeyDataset
from tqdm import tqdm

import sys
sys.path.append('Fine-Grained-Hallucination/GroundingDINO/groundingdino')
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionXLInpaintPipeline, DDIMScheduler, AutoencoderKL
from transformers import SamModel, SamProcessor
from PIL import Image
import numpy as np

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download

import nltk

from utils import get_entity_identifier_model, get_nouns, prepare_dino

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import re
from ultralytics import YOLO


device='cuda'

class MaskingDataset(Dataset):
    def __init__(self, img_dir, prompt_csv):
        self.img_dir = img_dir
        self.csv = prompt_csv
        self.df = pd.read_csv(self.csv)
        self.paths = [path for path in os.listdir(self.img_dir) if path.split('.')[0] in self.df['index'].apply(str).values]
        self.df = self.df.set_index('index')
        self.pattern = re.compile("[0-9]+")
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = os.path.join(self.img_dir, self.paths[idx])
        img = Image.open(path)
        img_name = int(self.pattern.findall(path)[-1])
        prompt = self.df.loc[img_name, "Prompts"]
        return {"image": img, "prompt": prompt, 'img_name': img_name}

def get_mask(image, prompt, dino_model, sam_model, sam_processor):
    global device
    
    transform = T.Compose(
    [
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    )
    image_source = np.asarray(image)
    image_transformed, _ = transform(image.convert('RGB'), None)
    boxes, logits, phrases = predict(
            model=dino_model, 
            image=image_transformed, 
            caption=prompt, 
            box_threshold=0.4, 
            text_threshold=0.9,
            device='cuda'
        )
    if len(boxes)>0:
        # boxes = boxes[0]
        input_boxes = []
        masks = {}
        for i, box in enumerate(boxes):
            x1 = ((box[0] - box[2]/2) * image_source.shape[1]).item()
            x2 = ((box[0] + box[2]/2) * image_source.shape[1]).item()
            y1 = ((box[1] - box[3]/2) * image_source.shape[0]).item()
            y2 = ((box[1] + box[3]/2) * image_source.shape[0]).item()
            inputs = sam_processor(image.convert('RGB'), input_boxes=[[[x1, y1, x2, y2]]], return_tensors='pt').to(device)
            with torch.no_grad():
                outputs = sam_model(**inputs)
            
            mask_out = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
            mask = mask_out[0].squeeze(0).permute(1,2,0).numpy()
            masks[f'{i+1}'] = Image.fromarray(mask[:,:,1])
        return masks
    else:
        return -1


def main():
    # sdxl_dataset_path = '/notebooks/Fine-Grained-Hallucination/sdxl_outputs'
    sdxl_dataset_path = '/notebooks/Fine-Grained-Hallucination/sd_2_outputs'
    prompt_csv = '/notebooks/Fine-Grained-Hallucination/target_prompt_dataset.csv'
    device = 'cuda'
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
    dino_model = prepare_dino()
    
    entity_pipeline = get_entity_identifier_model()
    
    sdxl_dataset = MaskingDataset(sdxl_dataset_path, prompt_csv)
    sdxl_dl = DataLoader(sdxl_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    # sd_2_dl = DataLoader(sd_2_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=False)
    
    if not os.path.exists(os.path.join(sdxl_dataset_path, "masks")):
        os.makedirs(os.path.join(sdxl_dataset_path, 'masks'))
        
    mask_dir = os.path.join(sdxl_dataset_path, 'masks')
    
    for i, batch in enumerate(sdxl_dl):
        prompt = batch[0]['prompt']
        nouns = get_nouns(prompt, entity_pipeline).split(',')
        short_prompt = prompt.replace(' ', "_") if len(prompt.replace(' ', "_"))<200 else prompt.replace(' ', "_")[:200]
        if not os.path.exists(os.path.join(mask_dir, str(batch[0]['img_name'])+'-'+short_prompt)):
            os.makedirs(os.path.join(mask_dir, str(batch[0]['img_name'])+'-'+short_prompt))
            
        this_img_mask_dir = os.path.join(mask_dir, str(batch[0]['img_name'])+'-'+short_prompt)
        
        for idx, noun in enumerate(nouns):
            path_noun = noun.replace(' ', "_") if " " in noun else noun
            if not os.path.exists(os.path.join(this_img_mask_dir, f"{path_noun}")):
                os.makedirs(os.path.join(this_img_mask_dir, f"{path_noun}"))
            this_noun_dir = os.path.join(this_img_mask_dir, f"{path_noun}")
            this_mask = get_mask(batch[0]['image'], noun, dino_model, sam_model, sam_processor)
            if not isinstance(this_mask, int):
                for k, v in this_mask.items():
                    v.save(f"{this_noun_dir}/{k}.png")
    
    yolo_model = YOLO("yolov8x.pt")
    
    for image_dir in os.listdir("./sd_2_outputs/masks"):
        image_id = image_dir.split('-')[0]
        for root, dirs, files in os.walk(os.path.join("./sd_2_outputs/masks", image_dir)):
            empty = True
            for subdir in dirs:
                subdir_path = os.path.join(root, subdir)
                if os.listdir(subdir_path):
                    empty = False
            if empty:
                result = yolo_model([f"./sd_2_outputs/{image_id}.jpg"])
                image = Image.open(f"./sd_2_outputs/{image_id}.jpg")
                boxes = result[0].boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes):
                    inputs = sam_processor(image.convert('RGB'), input_boxes=[[[box]]], return_tensors='pt').to(device)
                    with torch.no_grad():
                        outputs = sam_model(**inputs)

                    mask_out = sam_processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
                                                                                inputs["reshaped_input_sizes"].cpu())
                    mask = mask_out[0].squeeze(0).permute(1,2,0).numpy()
                    mask = Image.fromarray(mask[:,:,1])
                    mask.save(os.path.join(root, f'yolo_box_{i+1}.png'))
            break
            
if __name__=='__main__':
    main()