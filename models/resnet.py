import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Change to grayscale input
        self.model.conv1 = nn.Conv2d(
            1,64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Freeze early layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze last block (IMPORTANT for GradCAM)
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Train classifier
        self.model.fc = nn.Linear(512,2)

    def forward(self,x):

        return self.model(x)