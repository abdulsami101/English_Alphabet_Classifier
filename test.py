import train
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


custom_data_root = './dataset/'
batch_size = 2

# Transformer configuration
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize(size=(32,32)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading custom train dataset
custom_dataset = datasets.ImageFolder(root=os.path.join(custom_data_root, 'train_imgs'), transform=transform)

# Loading custom test dataset
custom_test_dataset = datasets.ImageFolder(root=os.path.join(custom_data_root, 'test_imgs'), transform=transform)

# Creating Dataloaders
test_loader = torch.utils.data.DataLoader(custom_test_dataset, batch_size=batch_size, shuffle=False)

# Define classes
Classes = custom_dataset.classes

# Device Configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model for inference
loaded_model = train.ConvNet()  # Create an instance of the ConvNet class
FILE = "eng_font.pth"
loaded_model.load_state_dict(torch.load(FILE))
loaded_model.to(device)
loaded_model.eval()  # Set the model for evaluation

# Here I'm using Testing using the loaded model
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(26)]
    n_class_samples = [0 for i in range(26)]

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = loaded_model(images)  # Use the loaded model here

        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the Test dataset: {acc} %')

    for i in range(26):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {Classes[i]}: {acc} %')
