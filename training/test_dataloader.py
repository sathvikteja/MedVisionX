import torch
import sys
import os

sys.path.append(os.path.abspath("."))
from torch.utils.data import DataLoader

from datasets.medical_dataset import MedicalDataset

train_data = MedicalDataset("data/processed/train")

loader = DataLoader(train_data,
                    batch_size=16,
                    shuffle=True)

for images,labels in loader:

    print("Batch shape:",images.shape)

    print("Labels:",labels)

    break