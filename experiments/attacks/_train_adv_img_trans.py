# MIT License
#
# This file is a copy from the "AttackVLM" project.
# See LICENSE.md for the full license text or visit the repo at:
# https://github.com/yunqing-me/AttackVLM/blob/main/LICENSE
#

import argparse
import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import random

from common.utils import FlatImageDatasetWithPaths
import numpy as np
import torch
import torch.backends.cudnn as cudnn
# import gradio as gr
import torchvision
from PIL import Image

from graphs.models.minigpt4.conversation.conversation import Chat, CONV_VISION_LLama2

# imports modules for registration
from common.config import Config
from common.registry import registry
from datasets.builders import *
from processors import blip_processors
from graphs.models import *
from graphs.models.minigpt4.common.optims import *

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


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

def load_finetuned_model(config, model):

        print("Loading finetuned VQAv2")
        checkpoint = config.model.vqa_finetuned                

        print(f"Loading checkpoint from {checkpoint}")
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
        

        print("Loading model state")
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("Loading model state. Done!")
        
        print(f"Numbers of treinable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    
class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index: int):
        original_tuple = super().__getitem__(index)  # (img, label)
        path, _ = self.samples[index]  # path: str

        image_processed = vis_processor(original_tuple[0])
        
        return image_processed, original_tuple[1], path

def main():
    parser = argparse.ArgumentParser()
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
    
    # parser.add_argument("--batch_size", default=10, type=int)
    # parser.add_argument("--num_samples", default=5000, type=int)
    # parser.add_argument("--alpha", default=1.0, type=float)
    # parser.add_argument("--epsilon", default=8, type=int)
    # parser.add_argument("--steps", default=100, type=int)
    # parser.add_argument("--output", default="/home/swf_developer/storage/attack/minigpt4_adv/", type=str, help='the folder name that restore your outputs')
    parser.add_argument("--batch_size", default=10, type=int)
    parser.add_argument("--num_samples", default=10, type=int)
    parser.add_argument("--alpha", default=1.0, type=float)
    parser.add_argument("--epsilon", default=8, type=int)
    parser.add_argument("--steps", default=100, type=int)
    parser.add_argument("--output", default="/home/swf_developer/storage/attack/minigpt4_adv/", type=str, help='the folder name that restore your outputs')
    args = parser.parse_args()

    alpha = args.alpha
    epsilon = args.epsilon

    # for normalized imgs
    scaling_tensor = torch.tensor((0.26862954, 0.26130258, 0.27577711), device=device)
    scaling_tensor = scaling_tensor.reshape((3, 1, 1)).unsqueeze(0)
    alpha = args.alpha / 255.0 / scaling_tensor
    epsilon = args.epsilon / 255.0 / scaling_tensor
    
    config = Config(args)
    print("Loading MiniGPT-4 models..")
         
    model_config = config.model
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        # load finetuned vqav2 checkpoint
    load_finetuned_model(config, model )

    vis_processor_cfg = config.datasets.evalvqav2.vis_processor.val
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
    print("Done")
    
    # ------------- pre-processing images/text ------------- #
    print("loading clean images")    
    imagenet_data = FlatImageDatasetWithPaths("/home/swf_developer/storage/attack/imagenet_clean_images/", transform=vis_processor)
        
    print("loading target images")    
    target_data   = FlatImageDatasetWithPaths("/home/swf_developer/storage/attack/targeted_images/samples", transform=vis_processor)
    
    print("loading dataloaders")
    data_loader_imagenet = torch.utils.data.DataLoader(imagenet_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    data_loader_target   = torch.utils.data.DataLoader(target_data, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    inverse_normalize = torchvision.transforms.Normalize(mean=[-0.48145466 / 0.26862954, -0.4578275 / 0.26130258, -0.40821073 / 0.27577711], std=[1.0 / 0.26862954, 1.0 / 0.26130258, 1.0 / 0.27577711])
    print("loading dataloaders.. done!")
    print("start attack")    

    # start attack
    try: 
        for i, ((image_org, path), (image_tgt, _)) in enumerate(zip(data_loader_imagenet, data_loader_target)):
            if args.batch_size * (i+1) > args.num_samples:                 
                break
            
            # (bs, c, h, w)
            image_org = image_org.to(device)
            image_tgt = image_tgt.to(device)
            
            # extract image features        
            with torch.no_grad():
                # tgt_image_features  -> size=(batch_size, 577, 768) ->
                # 577 tokens = 576 image patches + 1 [CLS] token
                # 768 is the embedding dimension                
                tgt_image_features = chat.forward_encoder(image_tgt) 
                
                # size=(batch_size, 768)
                # Select CLS token embedding as the image representation
                # select only the 0-th token (the [CLS] token)
                # select all 768 features of that token
                # the model’s learned representation of the whole image.
                tgt_image_features = (tgt_image_features)[:,0,:]                      

                # Computes the L2 norm of each feature vector
                # Normalize each embedding vector (useful for similarity comparisons)
                # Normalizes the tgt_image_features along dimension 1
                tgt_image_features = tgt_image_features / tgt_image_features.norm(dim=1, keepdim=True)
            
            # -------- get adv image -------- #        
            # requires_grad=True, PyTorch will track all operations involving delta and compute gradients for each element
            delta = torch.zeros_like(image_org, requires_grad=True)
            
            for j in range(args.steps):
                adv_image          = image_org + delta   # image is normalized to (0.0, 1.0)            
                adv_image_features = chat.forward_encoder(adv_image)            
                adv_image_features = adv_image_features[:,0,:]  # size = (bs, 768)
                adv_image_features = adv_image_features / adv_image_features.norm(dim=1, keepdim=True)

                # Calculates cosine similarity between the perturbed image and the target image
                # os_sim=∑(a⋅b)
                embedding_sim = torch.mean(torch.sum(adv_image_features * tgt_image_features, dim=1)) 
                embedding_sim.backward()
                            
                grad = delta.grad.detach()
                # Keeps it within a small, valid perturbation range [-ε, ε] so that it doesn't visibly distort the image.
                # torch.sign(grad) gives the direction to perturb each pixel.
                # alpha * torch.sign(grad) scales the direction by a small step size alpha.
                # min=-epsilon, max=epsilon ensures that all values in delta_data stay within the allowed perturbation limit.
                delta_data = torch.clamp(delta + alpha * torch.sign(grad), min=-epsilon, max=epsilon)
                delta.data = delta_data
                delta.grad.zero_()
                print(f"iter {i}/{args.num_samples//args.batch_size} step:{j:3d}, embedding similarity={embedding_sim.item():.5f}, max delta={torch.max(torch.abs(delta_data)).item():.3f}, mean delta={torch.mean(torch.abs(delta_data)).item():.3f}")

            # save imgs        
            adv_image = image_org + delta
            adv_image = torch.clamp(inverse_normalize(adv_image), 0.0, 1.0)
            
            for path_idx in range(len(path)):
                folder, name = path[path_idx].split("/")[-2], path[path_idx].split("/")[-1]            
                folder_to_save = os.path.join(args.output, folder)            
                if not os.path.exists(folder_to_save):
                    os.makedirs(folder_to_save, exist_ok=True)
                torchvision.utils.save_image(adv_image[path_idx], os.path.join(folder_to_save, name[:-4]) + '.png')
    except Exception as e: 
        print(e)
    
if __name__ == "__main__":
    main()