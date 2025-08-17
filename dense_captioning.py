from PIL import Image
import numpy as np
import os
import torch
import re

sdxl_dataset_path = '/mnt/data/workspace/misc/sdxl_outputs'
sd_2_dataset_path = '/mnt/data/workspace/misc/sd_2_outputs'
sdxl_mask_path = os.path.join(sdxl_dataset_path, 'masks')
sd_2_mask_path = os.path.join(sd_2_dataset_path, 'masks')

from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
blip_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
blip_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b").cuda()

from transformers import AutoModelForCausalLM, AutoTokenizer
meta_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
meta_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")

def get_centroid(mask):
    mask_array = np.array(mask)
    mask_coordinates = np.column_stack(np.where(mask_array > 0))
    if len(mask_coordinates) == 0:
        centroid = (None, None)
    else:
        # Calculate the median of the coordinates
        centroid = np.mean(mask_coordinates, axis=0)
        centroid = centroid[1], centroid[0]
    return tuple(centroid)

def get_dense_caption(mask, img):
    # prompt = "Describe the image with a focus on the intricate details of each object, including their colors, shapes, and numbers. Include any physical aspects that appear unusual or incorrect according to general knowledge."
    mask = np.array(mask)
    mask = mask[:, :, np.newaxis]
    mask = np.concatenate((mask, mask, mask), axis=2)
    img = np.array(img)
    img = mask * img
    img = Image.fromarray(img)
    prompt = 'Describe the image with a focus on the intricate details of the object, including their color, shape, and number. Include any physical aspects that appear unusual or incorrect according to general knowledge. You can ignore the pitch black background.'
    inputs = blip_processor(img, 
                            prompt, 
                            return_tensors="pt"
                            ).to("cuda", torch.float16)
    out = blip_model.generate(**inputs, max_length=200, do_sample=False)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def get_meta_caption(dense_captions):
    pattern = r'<caption>(.*?)</caption>'
    prompt = """I am providing you with captions for sub-regions of an image. These captions will be provided by the corresponding centroids
for the objects in the sub-regions. I want you to stitch all the dense captions into one unified caption for the entire image.
You have to use the centroid information to deduce the relative positions of each of the objects. Do not add any new information
to the captions. Make the caption as short as possible without losing too many details. Any mention of a black background should be ignored. Do not hallucinate any details. Generate the final caption within the <caption></caption> tags.

"""
    for i, this_caption in enumerate(dense_captions):
        string = f"{i+1}. {this_caption['centroid']} {this_caption['caption']}\n"
        prompt = prompt + string
    messages = [
    {"role": "system", "content": "You are a meta image captioning model. You look at various sub-captions and create a meaningful grounded caption using those. You can  use additional provided information to facilitate spatial reasoning."},
    {"role": "user", "content": prompt}
]
    text = meta_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
    model_inputs = meta_tokenizer([text], return_tensors="pt").to('cuda')
    generated_ids = meta_model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024
)
    generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
    response = meta_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    try:
        response = re.findall(pattern, response, re.DOTALL)[0]
        return response
    except:
        return response

from PIL import Image, ImageDraw
import pandas as pd
from tqdm import tqdm
df = pd.read_csv('/mnt/data/workspace/misc/DrawBenchPrompts.csv')
for image_name in tqdm(os.listdir(sdxl_dataset_path)):
    if not image_name.endswith('jpg'):
        continue
    idx = int(image_name.split('.')[0])
    image_path = os.path.join(sdxl_dataset_path, image_name)
    img = Image.open(image_path)
    mask_dir = [i for i in os.listdir(sdxl_mask_path) if idx==int(i.split('-')[0])][0]
    mask_dir = os.path.join(sdxl_mask_path, mask_dir)
    dense_captions = []
    for root, nouns, masks in os.walk(mask_dir):
        for mask in masks:
            mask_path = os.path.join(root, mask)
            mask = Image.open(mask_path)
            centroid = get_centroid(mask)
            caption = get_dense_caption(mask, img)
            description = {"centroid": centroid, "caption": caption}
            dense_captions.append(description)
    meta_caption = get_meta_caption(dense_captions)
    df.loc[idx, 'Meta Caption'] = meta_caption

df.to_csv('meta_captions_sdxl.csv', index=False)

df = pd.read_csv('/mnt/data/workspace/misc/DrawBenchPrompts.csv')
for image_name in tqdm(os.listdir(sd_2_dataset_path)):
    if not image_name.endswith('jpg'):
        continue
    idx = int(image_name.split('.')[0])
    image_path = os.path.join(sd_2_dataset_path, image_name)
    img = Image.open(image_path)
    mask_dir = [i for i in os.listdir(sd_2_mask_path) if idx==int(i.split('-')[0])][0]
    mask_dir = os.path.join(sd_2_mask_path, mask_dir)
    dense_captions = []
    for root, nouns, masks in os.walk(mask_dir):
        for mask in masks:
            mask_path = os.path.join(root, mask)
            mask = Image.open(mask_path)
            centroid = get_centroid(mask)
            caption = get_dense_caption(mask, img)
            description = {"centroid": centroid, "caption": caption}
            dense_captions.append(description)
    meta_caption = get_meta_caption(dense_captions)
    df.loc[idx, 'Meta Caption'] = meta_caption

df.to_csv('meta_captions_sd_2.csv', index=False)