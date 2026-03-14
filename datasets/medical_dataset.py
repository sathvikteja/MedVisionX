import os
import torch
from torch.utils.data import Dataset
import cv2

class MedicalDataset(Dataset):

    def __init__(self,root):

        self.paths = []
        self.labels = []

        classes = ["normal","tumor"]

        for label,cls in enumerate(classes):

            cls_path = os.path.join(root,cls)

            for file in os.listdir(cls_path):

                img_path = os.path.join(cls_path,file)

                self.paths.append(img_path)

                self.labels.append(label)

        # Optional debug mode (faster testing)
        # Uncomment if needed
        # self.paths = self.paths[:10000]
        # self.labels = self.labels[:10000]

    def __len__(self):

        return len(self.paths)


    def __getitem__(self,idx):

        img_path = self.paths[idx]

        image = cv2.imread(img_path,
                           cv2.IMREAD_GRAYSCALE)

        image = image/255.0

        image = torch.tensor(image).float()

        image = image.unsqueeze(0)

        label = torch.tensor(self.labels[idx])

        return image,label