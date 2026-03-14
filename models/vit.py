import torch
import torch.nn as nn
import timm

class ViTModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True,
            num_classes=2,
            in_chans=1
        )

    def forward(self,x):

        return self.model(x)