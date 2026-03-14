import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import cv2
import numpy as np
import glob
import timm

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

device = "cuda" if torch.cuda.is_available() else "cpu"

# LOAD MODEL
model = timm.create_model(
    "vit_small_patch16_224",
    pretrained=False,
    num_classes=2,
    in_chans=1
)

model.load_state_dict(
    torch.load(
        "results/vit_ssl_finetuned.pth",
        map_location=device
    )
)

model = model.to(device)
model.eval()

# ViT reshape function (IMPORTANT)
def reshape_transform(tensor):

    tensor = tensor[:,1:,:]

    height = width = int(np.sqrt(tensor.size(1)))

    result = tensor.reshape(
        tensor.size(0),
        height,
        width,
        tensor.size(2)
    )

    result = result.transpose(2,3).transpose(1,2)

    return result

# TARGET LAYER
target_layer = model.blocks[-1].norm1

# LOAD IMAGE
tumor_images = glob.glob("data/processed/test/tumor/*.png")

image_path = tumor_images[0]

print("Using image:", image_path)

image = cv2.imread(
    image_path,
    cv2.IMREAD_GRAYSCALE
)

image = cv2.resize(image,(224,224))

image_norm = image / 255.0

input_tensor = torch.tensor(
    image_norm,
    dtype=torch.float32
).unsqueeze(0).unsqueeze(0).to(device)

# GRADCAM
cam = GradCAM(
    model=model,
    target_layers=[target_layer],
    reshape_transform=reshape_transform
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

cv2.imwrite(
    "results/gradcam_vit.png",
    visualization
)

print("GradCAM saved")