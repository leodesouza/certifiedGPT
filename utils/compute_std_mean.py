import os
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm


def compute():
    root_dir = "E:/pesquisa_ia/projetos/datasets/vqav2/images/sample/val"

    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    num_pixels = 0
    pixel_sum = torch.zeros(3)  # 3 channels (RGB)
    pixels_sum_squared = torch.zeros(3)

    for root, _, files in os.walk(root_dir):
        for file in tqdm(files, desc="Processing images"):
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                img_path = os.path.join(root, file)
                image = Image.open(img_path).convert("RGB")
                tensor = transform(image)

                pixel_sum += tensor.sum(dim=(1, 2))  # Sum over H, W
                pixels_sum_squared += (tensor**2).sum(dim=(1, 2))  # Sum of squares over H, W
                num_pixels += tensor.size(1) * tensor.size(2)  # Total number of pixels

    mean = pixel_sum / num_pixels
    std = torch.sqrt(pixel_sum / num_pixels - mean**2)

    print(f"Mean: {mean.tolist()}")
    print(f"Std: {std.tolist()}")


if __name__ == "__main__":
    compute()
