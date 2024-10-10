# Vision Transformer Image Classifier from Scratch in Pytroch

This project demonstrates how to build an image classifier using a Vision Transformer (ViT) from scratch. The notebook takes the reader through the steps of creating a custom dataset, implementing key components of a Vision Transformer, training the model, and visualizing the results.

## Steps Followed:

### 1. Dataset Preparation
- The dataset is processed by converting images into patches, flattening the patch feature maps, and converting the output into the desired format for use in the Vision Transformer.
  
### 2. Patch Embedding Layer
- A patch embedding layer is created to transform image patches into a flattened representation. 

### 3. Transformer Encoder Layers
- Layer Normalization is applied using `torch.nn.LayerNorm()` to normalize input.
- Multi-Head Self-Attention (MSA) is implemented using `torch.nn.MultiheadAttention()`.
- A Multilayer Perceptron (MLP) block is created to handle the output of the self-attention layer.

### 4. Vision Transformer Construction
- A custom Transformer Encoder is created by combining the Layer Normalization, Multi-Head Self-Attention, and MLP blocks.

### 5. Training the Model
- The Vision Transformer model is trained on a custom dataset. 
- Results are evaluated, and the accuracy and loss curves are plotted.

### 6. Predictions
- After training, the model's performance is tested on new images.

### 7. Pretrained Model
- A pretrained model is loaded, and predictions are made for further comparison.

## Libraries Used:

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchinfo import summary
import matplotlib.pyplot as plt
import os
import requests
import torchvision
from helper_functions import plot_loss_curves
from going_modular.going_modular.predictions import pred_and_plot_image
from going_modular.going_modular import engine
