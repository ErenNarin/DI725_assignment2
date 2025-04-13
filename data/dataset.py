import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from transformers import DetrImageProcessor

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


class AUAIR(Dataset):
    def __init__(self, annotations, images_dir, transform=None):
        self.annotations = annotations
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = os.path.join(self.images_dir, ann["image_name"])
        image = Image.open(img_path).convert("RGB")

        width = ann["image_width:"]
        height = ann["image_height"]

        boxes = []
        labels = []

        for obj in ann["bbox"]:
            x_min = obj["left"]
            y_min = obj["top"]
            x_max = x_min + obj["width"]
            y_max = y_min + obj["height"]

            boxes.append([
                x_min,
                y_min,
                x_max,
                y_max,
            ])
            labels.append(obj["class"])

        target = {
            "class_labels": torch.tensor(labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float)
        }

        if self.transform:
            image = self.transform(image)

        return image, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    encoding = processor(images=images, return_tensors="pt")
    encoding["labels"] = list(targets)
    return encoding
