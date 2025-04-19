import torch
from torchvision.transforms import ToPILImage

import torch_xla
import torch_xla.core.xla_model as xm

toPilImage = ToPILImage()

def save_image(image, image_id, question_id, noise, output):
    
    imagem_cpu = image.cpu()
    imagem_cpu = imagem_cpu.squeeze(0) # removing batch dimension b. (B,C,W,H)
    
    #original image    
    pil = toPilImage(imagem_cpu)
    pil.save(f"{output}/{image_id}_{question_id}.png")
    
    #noisy image
    noisy_image = torch.clamp(imagem_cpu + noise, min=0, max=0)    
    pil = toPilImage(noisy_image)
    pil.save(f"{output}/{image_id}_{question_id}_noise_{int(noise * 100)}.png")
    