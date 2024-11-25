import os
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt


def open_file(image_id, question, answer):
    load_dotenv()
    vis_path = os.environ['DATA_DIR']
    vis_path = f"{vis_path}/images/train2014/all/"

    split = "train"

    file_name = f"COCO_{split}2014_{image_id:012d}.jpg"
    image_file_path = os.path.join(vis_path, file_name)
    image = Image.open(image_file_path).convert("RGB")

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f'Question: {question}\nAnswer: {answer}', fontsize=14)
    plt.show()
    plt.savefig('vqa_plot.png')


if __name__ == "__main__":
    open_file(35297, 'how many zebras?', 4)
