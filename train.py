import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 100
batch_size = 2
learning_rate = 0.001

# Custom Dataset root directory
custom_data_root = './dataset/'

# Transformer configuration
transform = transforms.Compose([
    # transforms.Resize(size=(32,32)),
    # transforms.Resize((128, 128), antialias=True),  # Add antialias=True
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Loading custom train dataset
custom_dataset = datasets.ImageFolder(root=os.path.join(custom_data_root, 'train_imgs'), transform=transform)


# Split the dataset into training and validation sets (80% train, 20% test)
train_size = 0.8 
train_dataset, validation_dataset = train_test_split(custom_dataset, train_size=train_size, test_size=1-train_size)

# Creating Dataloaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

# Define classes
Classes = custom_dataset.classes
print(Classes)
# Implement ConvNet
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


#main method
if __name__ == "__main__":

    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    print(n_total_steps)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward Pass 
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass and Optimize 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            print(f'EPOCH [{epoch + 1} / {num_epochs}], STEP [{i + 1} / {n_total_steps}], LOSS = {loss.item():.4f}')

    print('Finished Training')

    # Saving the Model
    FILE = "eng_font.pth"
    torch.save(model.state_dict(), FILE)