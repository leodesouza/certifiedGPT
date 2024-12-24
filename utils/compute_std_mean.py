import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Tensor()])


dataset = datasets.ImageFolder(root="path/to/coco_images", transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

num_pixels = 0
sum_channels = torch.zeros(3)  # 3 channels (RGB)
sum_channels_squared = torch.zeros(3)


for images, _ in dataloader:
    pixels = images.view(images.size(0), images.size(1), -1)  # Shape: [B, C, H*W]
    num_pixels += pixels.size(2) * images.size(0)  # Total number of pixels per channel

    sum_channels += pixels.sum(dim=(0, 2)).sum(dim=0)
    sum_channels_squared += (pixels * 2).sum(dim=(0, 2)).sum(dim=0)

mean = sum_channels / num_pixels
std = torch.sqrt(sum_channels_squared / num_pixels - mean**2)

print(f"Mean: {mean}")
print(f"Std: {std}")
