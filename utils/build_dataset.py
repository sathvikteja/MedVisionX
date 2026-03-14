import os
import nibabel as nib
import numpy as np
import cv2
from tqdm import tqdm

raw_path = "data/raw/brats"
output_path = "data/processed"

os.makedirs(output_path,exist_ok=True)

tumor_path = os.path.join(output_path,"tumor")
normal_path = os.path.join(output_path,"normal")

os.makedirs(tumor_path,exist_ok=True)
os.makedirs(normal_path,exist_ok=True)

patients = [p for p in os.listdir(raw_path)
            if os.path.isdir(os.path.join(raw_path,p))]

tumor_count = 0
normal_count = 0

for patient in tqdm(patients):

    patient_dir = os.path.join(raw_path,patient)

    flair_file = None
    seg_file = None

    for file in os.listdir(patient_dir):

        if "flair" in file:
            flair_file = os.path.join(patient_dir,file)

        if "seg" in file:
            seg_file = os.path.join(patient_dir,file)

    if flair_file is None or seg_file is None:
        continue

    flair = nib.load(flair_file).get_fdata()
    seg = nib.load(seg_file).get_fdata()

    for i in range(flair.shape[2]):

        img = flair[:,:,i]
        mask = seg[:,:,i]

        # Skip empty slices
        if img.max() == 0:
            continue

        # Normalize
        img = (img - img.min())/(img.max()-img.min())

        img = (img*255).astype(np.uint8)

        img = cv2.resize(img,(224,224))

        if mask.max()>0:

            filename = f"tumor_{tumor_count}.png"

            cv2.imwrite(os.path.join(tumor_path,filename),img)

            tumor_count+=1

        else:

            filename = f"normal_{normal_count}.png"

            cv2.imwrite(os.path.join(normal_path,filename),img)

            normal_count+=1

print("Tumor images:",tumor_count)
print("Normal images:",normal_count)