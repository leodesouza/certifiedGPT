import torch
# from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import to_pil_image

import torch_xla
import torch_xla.core.xla_model as xm

def unnormalize(image, mean, std):
    mean = torch.tensor(mean).view(-1,1,1) # reshape
    std = torch.tensor(std).view(-1,1,1)
    return image * std + mean

def save_image(image, image_id, question_id, noise, output):
    image_id = image_id.item()
    question_id = question_id.item()

    image_cpu = image.cpu()
    image_cpu = image_cpu.squeeze(0) # removing batch dimension b. (B,C,W,H)
    image_cpu = unnormalize(image_cpu, 0, 1)
    image_cpu = torch.clamp(image_cpu, min=0, max=1)        

    #original image    
    pil = to_pil_image(image_cpu)
    pil.save(f"{output}/{image_id}_{question_id}.png")
    
    #noisy image
    noisy_image = image_cpu + noise
    noisy_image = torch.clamp(noisy_image, min=0, max=1) # restrict tensor to min and max values 
    pil = to_pil_image(noisy_image)
    pil.save(f"{output}/{image_id}_{question_id}_noise_{int(noise * 100)}.png")
    