import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from datasets.medical_dataset import MedicalDataset
from models.vit import ViTModel

device = "cuda" if torch.cuda.is_available() else "cpu"

test_data = MedicalDataset("data/processed/test")

test_loader = DataLoader(
    test_data,
    batch_size=32
)

model = ViTModel().to(device)

model.load_state_dict(
    torch.load("results/vit_baseline.pth")
)

model.eval()

all_preds = []
all_labels = []

with torch.no_grad():

    for images,labels in test_loader:

        images = images.to(device)

        preds = model(images)

        _,predicted = torch.max(preds,1)

        all_preds.extend(predicted.cpu().numpy())

        all_labels.extend(labels.numpy())

print("Classification Report:")

print(
classification_report(
    all_labels,
    all_preds,
    target_names=["Normal","Tumor"]
)
)

print("Confusion Matrix:")

print(
confusion_matrix(
    all_labels,
    all_preds
)
)