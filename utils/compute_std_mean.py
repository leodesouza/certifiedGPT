import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


def compute():
    root_dir = "E:/pesquisa_ia/projetos/datasets/vqav2/images/sample/train"

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Tensor()])

    num_pixels = 0
    pixel_sum = torch.zeros(3)  # 3 channels (RGB)
    pixels_sum_squared = torch.zeros(3)

    for root, _, files in tqdm(files, desc="Processing images"):
        if file.lower().endswidth((".jpg", ".jpeg", ".png", ".bmp")):
            img_path = os.path.join(root, file)
            image = Image.open(img_path).convert("RGB")
            tensor = transform(image)

            pixel_sum += tensor.sum(dim=(1, 2))
            pixels_sum_squared += (tensor**2).sum(dim=(1, 2))
            num_pixels += tensor.size(1) * tensor.size(2)

    mean = pixel_sum / num_pixels
    std = torch.sqrt(pixel_sum / num_pixels - mean**2)

    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")


if __name__ == "__main__":
    compute()
