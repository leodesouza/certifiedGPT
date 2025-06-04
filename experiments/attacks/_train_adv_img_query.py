# MIT License
#
# This file is a copy from the "AttackVLM" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/yunqing-me/AttackVLM/blob/main/LICENSE
#

import argparse
import gc
import os
import random
import clip
from common.utils import FlatImageDatasetWithPaths
import numpy as np
import torch
import torchvision
from PIL import Image
import wandb
import copy
import time

from graphs.models.minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2, CONV_VISION_Vicuna0
# imports modules for registration
from common.config import Config
from common.registry import registry
from datasets.builders import *
from processors import blip_processors
from graphs.models import *
from graphs.models.minigpt4.common.optims import *


# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2023
device = "cuda" if torch.cuda.is_available() else "cpu"

# basic random seed
def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine
def seedEverything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)
# ------------------------------------------------------------------ #  

def to_tensor(pic):
    mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
    img = torch.from_numpy(np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True))
    img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
    img = img.permute((2, 0, 1)).contiguous()
    return img.to(dtype=torch.get_default_dtype())

transform = torchvision.transforms.Compose(
    [   
        torchvision.transforms.Lambda(lambda img: img.convert("RGB")),
        torchvision.transforms.Resize(size=(448, 448), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, max_size=None, antialias='warn'),
        torchvision.transforms.Lambda(lambda img: to_tensor(img)),
        # torchvision.transforms.ToTensor(),
    ]
)
normalize = torchvision.transforms.Compose(
    [   
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ]
)

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)
        path, _ = self.samples[index]
        return original_tuple + (path,)
    
def _i2t(args, chat, image_tensor):
    
    # normalize image here
    image_tensor = normalize(image_tensor / 255.0)
    
    
    conv = CONV_VISION_Vicuna0.copy()     
    num_beams = 1
    temperature = 1.0                               

    img_list = []                  
    chat.upload_img(image_tensor, conv, img_list)  # img embeddings, size() = [bs, 32, 5120]            
    chat.encode_img(img_list)  # img embeddings, size() = [bs, 32, 5120]                        
    chat.ask(args.query, conv)            
    
    # max_new_tokens=300,
    # max_length=2000
    captions, _  = chat.answer(conv, 
                            img_list, 
                            num_beams=num_beams, 
                            temperature=temperature)  
    return captions

def main():
    seedEverything()
    parser = argparse.ArgumentParser()
    # load models for i2t
    # minigpt-4
    parser.add_argument("--config-path", default="./configs/attack_configs/vqav2_eval_noise_0.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_samples", default=1, type=int)
    parser.add_argument("--input_res", default=448, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=5, type=int)
    parser.add_argument("--output", default="/home/swf_developer/storage/attack/query_based_attack_output/output.txt", type=str)
    parser.add_argument("--data_path", default="temp", type=str)
    parser.add_argument("--text_path", default="/home/swf_developer/storage/attack/img_2_txt_output/minigpt4_tmp_pred.txt", type=str)
    parser.add_argument("--query", default='what is the content of this image?', type=str)
    
    parser.add_argument("--delta", default="normal", type=str)
    parser.add_argument("--num_query", default=20, type=int)
    parser.add_argument("--num_sub_query", default=5, type=int)
    parser.add_argument("--sigma", default=8, type=float)
    
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project_name", type=str, default='certifiedgpt')
    parser.add_argument("--wandb_run_name", type=str, default='adv_img_query_attack_noise_0')
    
    args = parser.parse_args()


    config = Config(args)
    print("Loading MiniGPT-4 models..")
         
    model_config = config.model
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    vis_processor_cfg = config.datasets.evalvqav2.vis_processor.val
    vis_processor     = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)       
     
    # use clip text coder for attack
    clip_img_model_rn50,   _ = clip.load("RN50", device=device, jit=False)
    clip_img_model_rn101,  _ = clip.load("RN101", device=device, jit=False)
    clip_img_model_vitb16, _ = clip.load("ViT-B/16", device=device, jit=False)
    clip_img_model_vitb32, _ = clip.load("ViT-B/32", device=device, jit=False)
    clip_img_model_vitl14, _ = clip.load("ViT-L/14", device=device, jit=False)
    # ---------------------- #

    # load clip_model params
    num_sub_query, num_query, sigma = args.num_sub_query, args.num_query, args.sigma
    batch_size    = copy.deepcopy(args.batch_size)
    alpha         = args.alpha
    epsilon       = args.epsilon

    # load adv image
    # adv_vit_data      = ImageFolderWithPaths(args.data_path, transform=transform)
    adv_vit_data = FlatImageDatasetWithPaths("/home/swf_developer/storage/attack/imagenet_adv_images/images/", transform=transform)
    data_loader       = torch.utils.data.DataLoader(adv_vit_data, batch_size=batch_size, shuffle=False, num_workers=0)

    # load clean image
    clean_data        = FlatImageDatasetWithPaths("/home/swf_developer/storage/attack/imagenet_clean_images/", transform=transform)
    clean_data_loader = torch.utils.data.DataLoader(clean_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))     
    
    # org text/features
    adv_vit_text_path = args.text_path
    with open(os.path.join(adv_vit_text_path), 'r') as f:
        adv_vit_text  = f.readlines()[:args.num_samples] 
        f.close()
    
    # generating the text embeddings for org text/features 
    with torch.no_grad():
        adv_vit_text_token    = clip.tokenize(adv_vit_text, truncate=True).to(device)
        adv_vit_text_features = clip_img_model_vitb32.encode_text(adv_vit_text_token)
        adv_vit_text_features = adv_vit_text_features / adv_vit_text_features.norm(dim=1, keepdim=True)
        adv_vit_text_features = adv_vit_text_features.detach()
    
    # tgt text/features
    tgt_text_path = '/home/swf_developer/storage/attack/target/coco_captions_5000.txt'
    with open(os.path.join(tgt_text_path), 'r') as f:
        tgt_text  = f.readlines()[:args.num_samples] 
        f.close()
    
    # clip text features of the target
    # generating the text embeddings for tgt text/features
    with torch.no_grad():
        target_text_token    = clip.tokenize(tgt_text, truncate=True).to(device)
        target_text_features = clip_img_model_vitb32.encode_text(target_text_token)
        target_text_features = target_text_features / target_text_features.norm(dim=1, keepdim=True)
        target_text_features = target_text_features.detach()

    # baseline results
    vit_attack_results   = torch.sum(adv_vit_text_features * target_text_features, dim=1).squeeze().detach().cpu().numpy()
    query_attack_results = torch.sum(adv_vit_text_features * target_text_features, dim=1).squeeze().detach().cpu().numpy()
    assert (vit_attack_results == query_attack_results).all()
    
    ## other arch
    with torch.no_grad():
        # rn50
        print ("Checking similarity rn50")
        adv_vit_text_features_rn50 = clip_img_model_rn50.encode_text(adv_vit_text_token)
        adv_vit_text_features_rn50 = adv_vit_text_features_rn50 / adv_vit_text_features_rn50.norm(dim=1, keepdim=True)
        adv_vit_text_features_rn50 = adv_vit_text_features_rn50.detach()
        target_text_features_rn50  = clip_img_model_rn50.encode_text(target_text_token)
        target_text_features_rn50  = target_text_features_rn50 / target_text_features_rn50.norm(dim=1, keepdim=True)
        target_text_features_rn50  = target_text_features_rn50.detach()
        vit_attack_results_rn50    = torch.sum(adv_vit_text_features_rn50 * target_text_features_rn50, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_rn50  = torch.sum(adv_vit_text_features_rn50 * target_text_features_rn50, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_rn50 == query_attack_results_rn50).all()
        
        # rn101
        print ("Checking similarity rn101")
        adv_vit_text_features_rn101 = clip_img_model_rn101.encode_text(adv_vit_text_token)
        adv_vit_text_features_rn101 = adv_vit_text_features_rn101 / adv_vit_text_features_rn101.norm(dim=1, keepdim=True)
        adv_vit_text_features_rn101 = adv_vit_text_features_rn101.detach()
        target_text_features_rn101  = clip_img_model_rn101.encode_text(target_text_token)
        target_text_features_rn101  = target_text_features_rn101 / target_text_features_rn101.norm(dim=1, keepdim=True)
        target_text_features_rn101  = target_text_features_rn101.detach()
        vit_attack_results_rn101    = torch.sum(adv_vit_text_features_rn101 * target_text_features_rn101, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_rn101  = torch.sum(adv_vit_text_features_rn101 * target_text_features_rn101, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_rn101 == query_attack_results_rn101).all()
        

        # vitb16
        print ("Checking similarity vitb16")
        adv_vit_text_features_vitb16 = clip_img_model_vitb16.encode_text(adv_vit_text_token)
        adv_vit_text_features_vitb16 = adv_vit_text_features_vitb16 / adv_vit_text_features_vitb16.norm(dim=1, keepdim=True)
        adv_vit_text_features_vitb16 = adv_vit_text_features_vitb16.detach()
        target_text_features_vitb16  = clip_img_model_vitb16.encode_text(target_text_token)
        target_text_features_vitb16  = target_text_features_vitb16 / target_text_features_vitb16.norm(dim=1, keepdim=True)
        target_text_features_vitb16  = target_text_features_vitb16.detach()
        vit_attack_results_vitb16    = torch.sum(adv_vit_text_features_vitb16 * target_text_features_vitb16, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_vitb16  = torch.sum(adv_vit_text_features_vitb16 * target_text_features_vitb16, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_vitb16 == query_attack_results_vitb16).all()

        # vitl14
        print ("Checking similarity vitl14")
        adv_vit_text_features_vitl14 = clip_img_model_vitl14.encode_text(adv_vit_text_token)
        adv_vit_text_features_vitl14 = adv_vit_text_features_vitl14 / adv_vit_text_features_vitl14.norm(dim=1, keepdim=True)
        adv_vit_text_features_vitl14 = adv_vit_text_features_vitl14.detach()
        target_text_features_vitl14  = clip_img_model_vitl14.encode_text(target_text_token)
        target_text_features_vitl14  = target_text_features_vitl14 / target_text_features_vitl14.norm(dim=1, keepdim=True)
        target_text_features_vitl14  = target_text_features_vitl14.detach()
        vit_attack_results_vitl14    = torch.sum(adv_vit_text_features_vitl14 * target_text_features_vitl14, dim=1).squeeze().detach().cpu().numpy()
        query_attack_results_vitl14  = torch.sum(adv_vit_text_features_vitl14 * target_text_features_vitl14, dim=1).squeeze().detach().cpu().numpy()
        assert (vit_attack_results_vitl14 == query_attack_results_vitl14).all()
    ## ----------
    
    # if args.wandb:
    wandb.login(key=config.run.wandb_api_key)       
    wandb.init(project=args.wandb_project_name, name=args.wandb_run_name, reinit=True)                
    
    print("Start attack..")
    start_time = time.time()                
    for i, ((image, _), (image_clean, _)) in enumerate(zip(data_loader, clean_data_loader)):
        # if i % 50 != 0:
        #     continue

        if batch_size * (i+1) > args.num_samples:
            break

        print(f"{i}-th image")                
        image = image.to(device)  # size=(10, 3, args.input_res, args.input_res)
        image_clean = image_clean.to(device)  # size=(10, 3, 224, 224)
        
        # obtain all text features (via CLIP text encoder)
        adv_text_features = adv_vit_text_features[batch_size * (i): batch_size * (i+1)]        
        tgt_text_features = target_text_features[batch_size * (i): batch_size * (i+1)]
        
        # ------------------- random gradient-free method        
        delta = torch.tensor((image - image_clean))
        torch.cuda.empty_cache()
        
        best_caption = adv_vit_text[i]
        better_flag = 0
        
        # MF-tt
        for step_idx in range(args.steps):            
            # step 1. obtain purturbed images
            if step_idx == 0:
                image_repeat      = image.repeat(num_query, 1, 1, 1)  # size = (num_query x batch_size, 3, args.input_res, args.input_res)                
            else:
                image_repeat      = adv_image_in_current_step.repeat(num_query, 1, 1, 1)                             
                text_of_adv_image_in_current_step       = _i2t(args, chat, image_tensor=adv_image_in_current_step)
                adv_vit_text_token_in_current_step      = clip.tokenize(text_of_adv_image_in_current_step).to(device)
                adv_vit_text_features_in_current_step   = clip_img_model_vitb32.encode_text(adv_vit_text_token_in_current_step)
                adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step / adv_vit_text_features_in_current_step.norm(dim=1, keepdim=True)
                adv_vit_text_features_in_current_step   = adv_vit_text_features_in_current_step.detach()                
                adv_text_features     = adv_vit_text_features_in_current_step
                torch.cuda.empty_cache()
                
            # Rademacher noise distribution
            query_noise = torch.randn_like(image_repeat).sign()

            # limits the values in a tensor to a specified range.
            perturbed_image_repeat = torch.clamp(image_repeat + (sigma * query_noise), 0.0, 255.0)  # size = (num_query x batch_size, 3, args.input_res, args.input_res)
            
            # num_query is obtained via serveral iterations
            text_of_perturbed_imgs = []                       
            for query_idx in range(num_query//num_sub_query):
                sub_perturbed_image_repeat = perturbed_image_repeat[num_sub_query * (query_idx) : num_sub_query * (query_idx+1)]                                
                with torch.no_grad():
                    for i in range(sub_perturbed_image_repeat.size(0)):
                        img_tensor_i = sub_perturbed_image_repeat[i].unsqueeze(0)                         
                        text_i = _i2t(args, chat, image_tensor=img_tensor_i)
                        if isinstance(text_i, list):
                            text_of_perturbed_imgs.extend(text_i)
                        else:                    
                            text_of_perturbed_imgs.append(text_i)
                        
                        del img_tensor_i
                        del text_i
                        gc.collect()
                        torch.cuda.empty_cache()      
            
            # step 2. estimate grad
            with torch.no_grad():
                perturb_text_token    = clip.tokenize(text_of_perturbed_imgs, truncate=True).to(device)
                perturb_text_features = clip_img_model_vitb32.encode_text(perturb_text_token)
                perturb_text_features = perturb_text_features / perturb_text_features.norm(dim=1, keepdim=True)
                perturb_text_features = perturb_text_features.detach()

            # print("perturb_text_features size:", perturb_text_features.size())
            # print("adv_text_features size:", adv_text_features.size())
            # print("tgt_text_features size:", tgt_text_features.size())
            
            # computes a projection coefficient
            coefficient = torch.sum((perturb_text_features - adv_text_features) * tgt_text_features, dim=-1)
            coefficient = coefficient.reshape(num_query, batch_size, 1, 1, 1)
            query_noise = query_noise.reshape(num_query, batch_size, 3, args.input_res, args.input_res)
            pseudo_gradient = coefficient * query_noise / sigma 
            pseudo_gradient = pseudo_gradient.mean(0) 
            
            # step 3. log metrics
            delta_data = torch.clamp(delta + alpha * torch.sign(pseudo_gradient), min=-epsilon, max=epsilon)
            delta.data = delta_data
            # print(f"img: {i:3d}-step {step_idx} max  delta", torch.max(torch.abs(delta)).item())
            # print(f"img: {i:3d}-step {step_idx} mean delta", torch.mean(torch.abs(delta)).item())
            
            adv_image_in_current_step = torch.clamp(image_clean + delta, 0.0, 255.0)
            
            # log sim
            with torch.no_grad():
                text_of_adv_image_in_current_step = _i2t(args, chat, image_tensor=adv_image_in_current_step)
                text_token = clip.tokenize(text_of_adv_image_in_current_step, truncate=True).to(device)
                text_features_of_adv_image_in_current_step = clip_img_model_vitb32.encode_text(text_token)
                text_features_of_adv_image_in_current_step = text_features_of_adv_image_in_current_step / text_features_of_adv_image_in_current_step.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step = text_features_of_adv_image_in_current_step.detach()

                adv_txt_tgt_txt_score_in_current_step = torch.mean(torch.sum(text_features_of_adv_image_in_current_step * tgt_text_features, dim=1)).item()
                
                # update results
                print(f"query_attack_results.shape: {query_attack_results.shape}")
                
                if adv_txt_tgt_txt_score_in_current_step > query_attack_results[i]:
                    query_attack_results[i] = adv_txt_tgt_txt_score_in_current_step
                    best_caption = text_of_adv_image_in_current_step[0]
                    better_flag  = 1
                    
                # other clip archs
                # rn50
                tgt_text_features_rn50 = target_text_features_rn50[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_rn50 = clip_img_model_rn50.encode_text(text_token)
                text_features_of_adv_image_in_current_step_rn50 = text_features_of_adv_image_in_current_step_rn50 / text_features_of_adv_image_in_current_step_rn50.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_rn50 = text_features_of_adv_image_in_current_step_rn50.detach()
                adv_txt_tgt_txt_score_in_current_step_rn50 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_rn50 * tgt_text_features_rn50, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_rn50 > query_attack_results_rn50[i]:
                    query_attack_results_rn50[i] = adv_txt_tgt_txt_score_in_current_step_rn50
                
                # rn101
                tgt_text_features_rn101 = target_text_features_rn101[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_rn101 = clip_img_model_rn101.encode_text(text_token)
                text_features_of_adv_image_in_current_step_rn101 = text_features_of_adv_image_in_current_step_rn101 / text_features_of_adv_image_in_current_step_rn101.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_rn101 = text_features_of_adv_image_in_current_step_rn101.detach()
                adv_txt_tgt_txt_score_in_current_step_rn101 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_rn101 * tgt_text_features_rn101, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_rn101 > query_attack_results_rn101[i]:
                    query_attack_results_rn101[i] = adv_txt_tgt_txt_score_in_current_step_rn101
                
                # vitb16
                tgt_text_features_vitb16 = target_text_features_vitb16[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_vitb16 = clip_img_model_vitb16.encode_text(text_token)
                text_features_of_adv_image_in_current_step_vitb16 = text_features_of_adv_image_in_current_step_vitb16 / text_features_of_adv_image_in_current_step_vitb16.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_vitb16 = text_features_of_adv_image_in_current_step_vitb16.detach()
                adv_txt_tgt_txt_score_in_current_step_vitb16 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_vitb16 * tgt_text_features_vitb16, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_vitb16 > query_attack_results_vitb16[i]:
                    query_attack_results_vitb16[i] = adv_txt_tgt_txt_score_in_current_step_vitb16
                
                # vitl14
                tgt_text_features_vitl14 = target_text_features_vitl14[batch_size * (i): batch_size * (i+1)]
                text_features_of_adv_image_in_current_step_vitl14 = clip_img_model_vitl14.encode_text(text_token)
                text_features_of_adv_image_in_current_step_vitl14 = text_features_of_adv_image_in_current_step_vitl14 / text_features_of_adv_image_in_current_step_vitl14.norm(dim=1, keepdim=True)
                text_features_of_adv_image_in_current_step_vitl14 = text_features_of_adv_image_in_current_step_vitl14.detach()
                adv_txt_tgt_txt_score_in_current_step_vitl14 = torch.mean(torch.sum(text_features_of_adv_image_in_current_step_vitl14 * tgt_text_features_vitl14, dim=1)).item()
                if adv_txt_tgt_txt_score_in_current_step_vitl14 > query_attack_results_vitl14[i]:
                    query_attack_results_vitl14[i] = adv_txt_tgt_txt_score_in_current_step_vitl14
                    # ----------------                            
        
        wandb.log(
            {   
                "moving-avg-adv-rn50"    : np.mean(vit_attack_results_rn50[:(i+1)]),
                "moving-avg-query-rn50"  : np.mean(query_attack_results_rn50[:(i+1)]),
                
                "moving-avg-adv-rn101"   : np.mean(vit_attack_results_rn101[:(i+1)]),
                "moving-avg-query-rn101" : np.mean(query_attack_results_rn101[:(i+1)]),
                
                "moving-avg-adv-vitb16"  : np.mean(vit_attack_results_vitb16[:(i+1)]),
                "moving-avg-query-vitb16": np.mean(query_attack_results_vitb16[:(i+1)]),
                
                "moving-avg-adv-vitb32"  : np.mean(vit_attack_results[:(i+1)]),
                "moving-avg-query-vitb32": np.mean(query_attack_results[:(i+1)]),
                
                "moving-avg-adv-vitl14"  : np.mean(vit_attack_results_vitl14[:(i+1)]),
                "moving-avg-query-vitl14": np.mean(query_attack_results_vitl14[:(i+1)]),
            }
        )

        
        # log text
        print("best caption of current image:", best_caption)
        with open(os.path.join(args.output), 'a') as f:
            # print(''.join([best_caption]), file=f)
            if better_flag:
                f.write(best_caption+'\n')
            else:
                f.write(best_caption)
        f.close()
    end_time = time.time()
    duration = (end_time - start_time) / 60
    # print(f"{step_idx}-th step ended. {duration:2f} minutes")    
    print(f"attack ended. {duration:2f} minutes")    
if __name__ == "__main__":
    main()