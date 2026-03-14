import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm

from datasets.ssl_dataset import SSLDataset


def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:",device)

    dataset = SSLDataset(
        "data/processed/train"
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True
    )

    model = timm.create_model(
        "vit_small_patch16_224",
        pretrained=True,
        num_classes=4,
        in_chans=1
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001
    )

    epochs = 2

    for epoch in range(epochs):

        total_loss = 0

        model.train()

        for batch_idx,(images,labels) in enumerate(loader):

            images = images.to(device)
            labels = labels.to(device)

            preds = model(images)

            loss = criterion(preds,labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

            if batch_idx%200==0:

                print(
                    f"Batch {batch_idx} Loss {loss.item():.4f}"
                )

        print(
            f"Epoch {epoch} Loss {total_loss/len(loader)}"
        )

    torch.save(
        model.state_dict(),
        "results/vit_ssl_pretrained.pth"
    )

    print("SSL training complete")


if __name__ == "__main__":

    main()