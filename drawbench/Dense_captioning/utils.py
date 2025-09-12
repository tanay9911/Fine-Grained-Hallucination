import sys
sys.path.append('/mnt/data/Grounded-Segment-Anything/GroundingDINO/groundingdino')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
torch.random.manual_seed(0)

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download

def get_entity_identifier_model():
    model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct", 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
    pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
    return pipe

def prepare_dino():
    cache_config_file = hf_hub_download(repo_id="ShilongLiu/GroundingDINO", filename="GroundingDINO_SwinB.cfg.py")

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = 'cuda'

    cache_file = hf_hub_download(repo_id="ShilongLiu/GroundingDINO", filename="groundingdino_swinb_cogcoor.pth")
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    model.cuda()
    _ = model.eval()
    return model 

def get_nouns(prompt, pipe):
    messages = [
        {"role": "user", "content": f"Can you provide all the noun phrases in the given sentence? Make sure to return the answers separated by commas. Only extract phrases which contain a noun, not just adjectives or prepositions. Make sure each phrase has a unique noun, try not to repeat nouns. Do not rephrase the answers, extract and output them as given in the sentence. If you are unable to find any nouns, just repeat the original sentence.\n{prompt}"}
    ]
    generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False
    }
    output = pipe(messages, **generation_args)
    return output[0]['generated_text']

