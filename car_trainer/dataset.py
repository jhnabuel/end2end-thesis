import json
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch

class SelfDrivingDataset(Dataset):
    def __init__(self, catalog_path, base_image_dir):
        self.base_image_dir = base_image_dir
        self.samples = []

        with open(catalog_path, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)

                    if record['cam/image_array']:
                        self.samples.append(record)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        record = self.samples[index]

        image_path = os.path.join(self.base_image_dir, record['cam/image_array'])
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (200, 66))

        image_tensor = torch.from_numpy(img).float().permute(2, 0, 1)

        normalized_steering_angle = float(record['angle']) / 50.0
        label_tensor = torch.tensor([normalized_steering_angle], dtype=torch.float32)

        return image_tensor, label_tensor


if __name__ == "__main__":
    base_image_dir = "../data"
    catalog_path = "../data/catalog_2026-03-24.catalog"


    dataset = SelfDrivingDataset(catalog_path=catalog_path, base_image_dir=base_image_dir)

    print("Datset size: ", len(dataset))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_images, batch_labels in dataloader:
        print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")
    image, label = dataset[0]

    print("Image shape:", image.shape)   # should be [3, 66, 200]
    print("Label:", label)

    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
