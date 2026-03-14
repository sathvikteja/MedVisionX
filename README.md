# MedVisionX: Brain Tumor Classification using Vision Transformers and Self-Supervised Learning

## Project Overview

MedVisionX is a deep learning project focused on brain tumor classification from MRI scans using modern computer vision techniques. The project compares CNNs and Vision Transformers and explores self-supervised learning (SSL) for representation learning.

The goal is to evaluate whether transformer-based architectures outperform traditional convolutional networks for medical imaging tasks and to study the effect of SSL on feature learning.

---

## Key Features

• CNN baseline using ResNet18  
• Vision Transformer baseline  
• Self-Supervised Learning (SSL) pretraining  
• Transfer learning fine-tuning  
• Model explainability using GradCAM  
• Performance comparison across architectures  
• Medical AI focused evaluation pipeline  

---

## Dataset

Brain MRI dataset containing two classes:

• Normal  
• Tumor  

Dataset pipeline includes:

• Data preprocessing  
• Train / validation / test split  
• Image normalization  
• Custom PyTorch Dataset and DataLoader  

Dataset is not included due to size constraints.

---

## Model Architectures

### ResNet18 Baseline

Traditional CNN architecture used as a baseline model.

Advantages:
• Strong spatial feature extraction  
• Proven performance in medical imaging  

Limitations:
• Limited global context understanding  

---

### Vision Transformer (ViT)

Transformer-based architecture using patch embeddings and self-attention.

Advantages:
• Global attention mechanism  
• Better spatial reasoning  
• Strong generalization capability  

---

### Self-Supervised Vision Transformer

SSL pretraining performed before supervised fine-tuning.

Pipeline:

SSL Pretraining → Representation Learning  
Fine-tuning → Tumor Classification  

Purpose:

Improve feature representations without requiring labels.

---

## Training Pipeline

Project workflow:

Dataset → Preprocessing → Model Training → Evaluation → Explainability

Steps:

1. Train CNN baseline  
2. Train ViT baseline  
3. SSL pretraining  
4. SSL fine-tuning  
5. Model evaluation  
6. GradCAM analysis  

---

## Results

### Model Performance Comparison

| Model | Accuracy |
|-------|----------|
| ResNet18 | 83% |
| Vision Transformer | 96% |
| SSL Vision Transformer | 96% |

---

## Key Observations

• Vision Transformers significantly outperform CNNs  
• Transformer attention captures tumor structure better  
• SSL improved representation quality  
• Dataset size likely limited SSL gains  
• ViT shows strong generalization ability  

---

## Evaluation Metrics

Metrics used:

• Accuracy  
• Precision  
• Recall  
• F1 Score  
• Confusion Matrix  

Final SSL ViT Results:

Accuracy: **96%**

---

## Explainability

GradCAM was used to visualize model attention regions.

Generated:

• ResNet GradCAM  
• ViT GradCAM  

Purpose:

Verify that models focus on tumor regions instead of irrelevant features.

Results stored in:
results/
gradcam_resnet.png
gradcam_vit.png
vit_ssl_confusion_matrix.png


---

## Project Structure


MedVisionX/

datasets/
models/
training/
evaluation/
explainability/
utils/

results/
gradcam_resnet.png
gradcam_vit.png
vit_ssl_confusion_matrix.png
final_results.txt

requirements.txt
main.py
README.md


---

## Installation

Clone repository:


git clone https://github.com/YOUR_USERNAME/MedVisionX.git

cd MedVisionX


Install dependencies:


pip install -r requirements.txt


---

## Training

Train CNN baseline:


python training/train_resnet.py


Train Vision Transformer:


python training/train_vit.py


SSL pretraining:


python training/train_ssl_vit.py


SSL fine-tuning:


python training/train_ssl_finetune_vit.py


---

## Evaluation

Evaluate ViT:


python evaluation/evaluate_vit.py


---

## Explainability

Generate GradCAM:


python explainability/gradcam_resnet.py

python explainability/gradcam_vit.py


---

## Technologies Used

Python  
PyTorch  
Torchvision  
TIMM  
NumPy  
OpenCV  
Scikit-learn  
GradCAM  

---

## Key Learnings

This project demonstrates:

• Transformer advantages over CNNs  
• Self-supervised learning workflow  
• Transfer learning strategies  
• Medical AI evaluation practices  
• Model interpretability techniques  

---

## Future Improvements

Possible extensions:

• Larger medical datasets  
• Multi-class tumor classification  
• Swin Transformers  
• DINO self-supervised learning  
• MONAI medical imaging framework  
• Clinical deployment pipeline  

---

## Author

Teja  

Machine Learning | Computer Vision | AI  

---

## Notes

Model weights and dataset are excluded due to size limitations.
