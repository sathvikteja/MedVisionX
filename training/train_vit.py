import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.medical_dataset import MedicalDataset
from models.vit import ViTModel


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:",device)

    train_data = MedicalDataset("data/processed/train")
    val_data = MedicalDataset("data/processed/val")

    train_loader = DataLoader(
        train_data,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_data,
        batch_size=32
    )

    model = ViTModel().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    epochs = 2

    best_acc = 0

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch_idx,(images,labels) in enumerate(train_loader):

            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)

            loss = criterion(preds,labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 200 == 0:

                print(f"Batch {batch_idx} Loss {loss.item():.4f}")

        train_loss = total_loss/len(train_loader)

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for images,labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                preds = model(images)

                _,predicted = torch.max(preds,1)

                total += labels.size(0)

                correct += (predicted==labels).sum().item()

        val_acc = 100*correct/total

        print("---------------")
        print(f"Epoch {epoch}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.2f}%")

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                "results/vit_baseline.pth"
            )

            print("Best model saved")

    print("Training complete")


if __name__ == "__main__":

    main()