# MIT License
#
# This file is a copy from the "AttackVLM" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/yunqing-me/AttackVLM/blob/main/LICENSE
#

import argparse
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
import io
import random
from PIL import Image
import time

from common.utils import FlatImageDatasetWithPaths
import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn


from graphs.models.minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2, CONV_VISION_Vicuna0, Conversation, SeparatorStyle

# imports modules for registration
# imports modules for registration
from common.config import Config
from common.registry import registry
from datasets.builders import *
from processors import blip_processors
from graphs.models import *
from graphs.models.minigpt4.common.optims import *


# seed for everything
# credit: https://www.kaggle.com/code/rhythmcam/random-seed-everything
DEFAULT_RANDOM_SEED = 2025
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

def load_finetuned_model(config, model):
    print("Loading finetuned VQAv2")
    # checkpoint = config.model.vqa_finetuned
    # chk_path = "/home/swf_developer/storage/checkpoints/certifiedgpt/vqav2_finetuning_noise_0/vqav2_finetuning_with_optim_noise_0.pth"
    chk_path = "/home/swf_developer/storage/checkpoints/certifiedgpt/vqav2_finetuning_noise_0.25/vqav2_finetuning_with_optim_noise_0.25.pth"    
        
    checkpoint = torch.load(chk_path, map_location=torch.device('cpu'))

    print("Loading model state")
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print("Loading model state. Done!")

    print(f"Numbers of treinable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
# ------------------------------------------------------------------ #  

def main():
    seedEverything()
    parser = argparse.ArgumentParser(description="Demo")
    
    # minigpt-4
    parser.add_argument("--config-path", default="./configs/certify_configs/vqav2_certify_noise_0.25.yaml", help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    
    # obtain text in batch
    parser.add_argument("--img_file", default='/raid/common/imagenet-raw/val/n01440764/ILSVRC2012_val_00003014.png', type=str)
    # parser.add_argument("--img_path", default='/home/swf_developer/storage/attack/imagenet_adv_images/images/', type=str)
    parser.add_argument("--img_path", default='/home/swf_developer/storage/attack/imagenet_clean_images/', type=str)
    # parser.add_argument("--query", default='[vqa] Respond to this question in English with a short answer: what is the content of this image? ', type=str)
    parser.add_argument("--query", default='[vqa] What is shown in the image? ', type=str)
        
    parser.add_argument("--output_path", default="minigpt4_tmp", type=str)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--batch_size_in_gen", default=3, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    args = parser.parse_args()
    
    
    config = Config(args)
    print("Loading MiniGPT-4 models..")
         
    model_config = config.model
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

    load_finetuned_model(config, model)
    
    vis_processor_cfg = config.datasets.evalvqav2.vis_processor.val
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)    
            

    # load image
    print(f"loading images from path: {args.img_path}")
    imagenet_data = FlatImageDatasetWithPaths(args.img_path, transform=vis_processor)
    dataloader    = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    
    # img2txt
    print("start iteration...")    
    for i, (image, _) in enumerate(dataloader):
        start = time.perf_counter()
        
        print(f"MiniGPT4 img2txt: {i}/{args.num_samples//args.batch_size}")
        if i >= args.num_samples//args.batch_size:
            print(f"Successfully processed {args.num_samples} images to text!")
            break 

        model.eval() 
        with torch.no_grad():
            # conv = CONV_VISION_Vicuna0.copy()                                    

            # img_list = []      

            # print("up load imgs")      
            # chat.upload_img(image, conv, img_list)  # img embeddings, size() = [bs, 32, 5120]

            # print("econde imgs")      
            # chat.encode_img(img_list)  # img embeddings, size() = [bs, 32, 5120]            

            # print("ask to minigpt4")                              
            # chat.ask(args.query, conv)            

            # print("answer...")      
            # captions, _  = chat.answer(conv, 
            #                         img_list, 
            #                         num_beams=num_beams, 
            #                         temperature=temperature,
            #                         max_new_tokens=20,
            #                         max_length=2000)
            # print(f"caption: {captions}")

             # Removed `xla_amp.autocast` and used PyTorch's native autocast
            

            # instruction = f"[vqa] Based on the image, respond to this question in English with a short answer: {args.query}"
            instruction = "<Img><ImageHere></Img> {} ".format(args.query)
        
            print(f"INSTRUCTION: {instruction}")
            # with torch.cuda.amp.autocast(enabled=config.run.amp):
            captions, _ = model.generate(
                image, [instruction], max_new_tokens=config.run.max_new_tokens, do_sample=False, calc_probs=False
            )
                
            print(f"caption ---> : {captions}")
            raise ValueError("stop")

            # img_list   = chat.get_img_list(image)
            # mixed_embs = chat.get_mixed_embs(args, img_list=img_list)
            # captions   = chat.get_text(mixed_embs)

        # write captions
        # with open(os.path.join("/home/swf_developer/storage/attack/img_2_txt_output", args.output_path + '_pred.txt'), 'a') as f:
        #     print('\n'.join(captions), file=f)
        # f.close()
        
        end = time.perf_counter()
        print(f"query time for {args.batch_size} samples:", (end - start))
        
    print("Caption saved.")

if __name__ == "__main__":
    main()