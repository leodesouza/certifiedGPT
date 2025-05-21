# MIT License
#
# This file is a copy from the "AttackVLM" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/yunqing-me/AttackVLM/blob/main/LICENSE
#

import argparse
import os
import random
from PIL import Image
import time

from common.utils import FlatImageDatasetWithPaths
import numpy as np
import torch
import torchvision
import torch.backends.cudnn as cudnn

from common.config import Config
from common.registry import registry

from minigpt4.common.registry import registry
from graphs.models.minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


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


class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]  # path: str

        image_processed = vis_processor(original_tuple[0])
        return image_processed, original_tuple[1], path
    
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
    parser.add_argument("--img_path", default='/home/swf_developer/storage/attack/imagenet_adv_images/images/', type=str)
    parser.add_argument("--query", default='what is the content of this image?', type=str)
    
    parser.add_argument("--output_path", default="minigpt4_tmp", type=str)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--batch_size_in_gen", default=3, type=int)
    parser.add_argument("--num_samples", default=1000, type=int)
    args = parser.parse_args()
    
    
    config = Config(args)
    print("Loading MiniGPT-4 models..")
         
    model_config = config.model
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    
    vis_processor_cfg = config.datasets.evalvqav2.vis_processor.val
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)    
    num_beams = 1
    temperature = 1.0
    print("Done.")

    # load image
    imagenet_data = FlatImageDatasetWithPaths(args.img_path, transform=vis_processor)
    dataloader    = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=24)

    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    conv = CONV_VISION_LLama2.copy()
    
    # img2txt
    for i, (image, _, path) in enumerate(dataloader):
        start = time.perf_counter()
        
        print(f"MiniGPT4 img2txt: {i}/{args.num_samples//args.batch_size}")
        if i >= args.num_samples//args.batch_size:
            print(f"Successfully processed {args.num_samples} images to text!")
            break 
        image = image.to(device)
        with torch.no_grad():
            img_list = []            
            chat.upload_img(image, conv, img_list)  # img embeddings, size() = [bs, 32, 5120]
            chat.encode_img(img_list)  # img embeddings, size() = [bs, 32, 5120]
            # mixed_embs = chat.get_mixed_embs(args, img_list=img_list)
            captions   = chat.answer(args, conv, img_list)
        # write captions
        with open(os.path.join("/home/swf_developer/storage/attack/img_2_txt_output", args.output_path + '_pred.txt'), 'a') as f:
            print('\n'.join(captions), file=f)
        f.close()
        
        end = time.perf_counter()
        print(f"query time for {args.batch_size} samples:", (end - start))
        
    print("Caption saved.")
        
if __name__ == "__main__":
    main()