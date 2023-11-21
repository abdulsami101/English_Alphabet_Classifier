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
from PIL import Image

custom_data_root = './dataset/'

# Transformer configuration
transform = transforms.Compose([
    # transforms.Resize(size=(32,32)),
    # transforms.Resize((128, 128), antialias=True),  # Add antialias=True
   
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading custom train dataset
custom_dataset = datasets.ImageFolder(root=os.path.join(custom_data_root, 'train_imgs'), transform=transform)

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



# Function to predict the class of a single image
def predict_single_image(image_path):
    # Load the image
    img = Image.open(image_path)

    # Apply the same transformations as during training and testing
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension

    # Move the image to the device
    img = img.to(device)

    # Perform inference
    with torch.no_grad():
        output = loaded_model(img)
        _, predicted_class = torch.max(output, 1)

    # Get the predicted class label
    predicted_label = Classes[predicted_class.item()]

    print(f'The predicted class for the image is: {predicted_label}')

# Provide the path to the image you want to classify
image_path = './4_9.png'

# Call the function with the image path
predict_single_image(image_path)
