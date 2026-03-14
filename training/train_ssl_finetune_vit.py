import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets.medical_dataset import MedicalDataset
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

train_dataset = MedicalDataset("data/processed/train")
val_dataset = MedicalDataset("data/processed/val")

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32
)

# Create model (grayscale input)
model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=2,
    in_chans=1
)

# LOAD SSL WEIGHTS
ssl_weights = torch.load(
    "results/vit_ssl_pretrained.pth",
    map_location=torch.device(device)
)

# REMOVE incompatible layers (IMPORTANT)
ssl_weights.pop("patch_embed.proj.weight", None)
ssl_weights.pop("patch_embed.proj.bias", None)
ssl_weights.pop("head.weight", None)
ssl_weights.pop("head.bias", None)

# Load encoder weights only
model.load_state_dict(
    ssl_weights,
    strict=False
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-5
)

best_acc = 0

for epoch in range(5):

    model.train()

    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print("---------------")
    print("Epoch:", epoch)
    print("Train Loss:", total_loss / len(train_loader))

    # VALIDATION
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total

    print("Val Accuracy:", acc)

    if acc > best_acc:

        best_acc = acc

        torch.save(
            model.state_dict(),
            "results/vit_ssl_finetuned.pth"
        )

        print("Best SSL model saved")

print("SSL Fine-tuning complete")