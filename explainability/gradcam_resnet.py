import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import cv2
import numpy as np
import glob

from torchvision import models
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)

# CREATE MODEL
model = models.resnet18(weights=None)

model.conv1 = torch.nn.Conv2d(
    1,
    64,
    kernel_size=7,
    stride=2,
    padding=3,
    bias=False
)

model.fc = torch.nn.Linear(512,2)

# LOAD WEIGHTS
state_dict = torch.load(
    "results/resnet_baseline.pth",
    map_location=device
)

# REMOVE "model." PREFIX
new_state_dict = {}

for k,v in state_dict.items():

    new_key = k.replace("model.","")

    new_state_dict[new_key] = v

model.load_state_dict(new_state_dict)

model = model.to(device)

model.eval()

# TARGET LAYER
target_layer = model.layer4[-1]

# PICK ANY TUMOR IMAGE
tumor_images = glob.glob("data/processed/test/tumor/*.png")

image_path = tumor_images[0]

print("Using image:", image_path)

# LOAD IMAGE
image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

if image is None:

    raise Exception("Image not found")

image = cv2.resize(image,(224,224))

image_norm = image / 255.0

input_tensor = torch.tensor(
    image_norm,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)

# GRADCAM
cam = GradCAM(
    model=model,
    target_layers=[target_layer]
)

grayscale_cam = cam(
    input_tensor=input_tensor
)

grayscale_cam = grayscale_cam[0]

# CONVERT FOR VISUALIZATION
image_rgb = cv2.cvtColor(
    (image_norm*255).astype(np.uint8),
    cv2.COLOR_GRAY2RGB
)

image_rgb = image_rgb / 255.0

visualization = show_cam_on_image(
    image_rgb,
    grayscale_cam,
    use_rgb=True
)

# SAVE RESULT
cv2.imwrite(
    "results/gradcam_resnet.png",
    visualization
)

print("ResNet GradCAM saved")