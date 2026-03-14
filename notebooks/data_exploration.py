import os
import nibabel as nib
import matplotlib.pyplot as plt

patient = "BraTS20_Training_001"

base_path = "data/raw/brats"

patient_path = os.path.join(base_path, patient)

flair_path = os.path.join(patient_path,
                          patient+"_flair.nii")

seg_path = os.path.join(patient_path,
                        patient+"_seg.nii")

flair = nib.load(flair_path).get_fdata()
seg = nib.load(seg_path).get_fdata()

print("MRI shape:", flair.shape)
print("Mask shape:", seg.shape)

slice_idx = flair.shape[2]//2

mri_slice = flair[:,:,slice_idx]
mask_slice = seg[:,:,slice_idx]

plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.imshow(mri_slice,cmap='gray')
plt.title("MRI")

plt.subplot(1,2,2)
plt.imshow(mri_slice,cmap='gray')

# overlay tumor
plt.imshow(mask_slice,
           cmap='jet',
           alpha=0.5)

plt.title("Tumor Overlay")

plt.show()

# Count tumor slices

tumor_slices = 0
normal_slices = 0

for i in range(flair.shape[2]):

    mask_slice = seg[:,:,i]

    if mask_slice.max() > 0:
        tumor_slices += 1
    else:
        normal_slices += 1

print("Total slices:", flair.shape[2])
print("Tumor slices:", tumor_slices)
print("Normal slices:", normal_slices)