import os
import torch
from torch.utils.data import Dataset
import cv2
import random

class SSLDataset(Dataset):

    def __init__(self,root):

        self.paths = []

        classes = ["normal","tumor"]

        for cls in classes:

            cls_path = os.path.join(root,cls)

            for file in os.listdir(cls_path):

                self.paths.append(
                    os.path.join(cls_path,file)
                )

    def __len__(self):

        return len(self.paths)

    def __getitem__(self,idx):

        img_path = self.paths[idx]

        image = cv2.imread(
            img_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = image/255.0

        # Random rotation
        rot = random.randint(0,3)

        image = torch.tensor(image).float()

        image = image.unsqueeze(0)

        image = torch.rot90(
            image,
            rot,
            dims=[1,2]
        )

        label = torch.tensor(rot)

        return image,label