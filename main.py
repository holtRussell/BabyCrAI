import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader




# Utility function to convert audio to mel-spectrogram image
def audio_to_melspectrogram(audio_path, save_path, sr=16000, n_mels=128, n_fft=2048, hop_length=512):
    print(f"Converting audio to mel-spectrogram: {audio_path}")
    y, sr = librosa.load(audio_path, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved mel-spectrogram image: {save_path}")


# Paths
audio_dir = 'data/donate_a_cry/donateacry_corpus'  # Update with actual path
image_dir = 'data/donate_a_cry/melspectrogram_images'  # Update with actual path
print(f"Creating image directory: {image_dir}")
os.makedirs(image_dir, exist_ok=True)

# Parameters
sr = 16000
n_mels = 128
n_fft = 2048
hop_length = 512
classes = ['hungry', 'burping', 'discomfort', 'belly_pain', 'tired']
print(f"Classes defined: {classes}")

# Convert audio files to mel-spectrogram images
for class_name in classes:
    class_audio_dir = os.path.join(audio_dir, class_name)
    class_image_dir = os.path.join(image_dir, class_name)
    print(f"Creating class image directory: {class_image_dir}")
    os.makedirs(class_image_dir, exist_ok=True)

    for audio_file in os.listdir(class_audio_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(class_audio_dir, audio_file)
            image_name = audio_file.replace('.wav', '.jpg')
            image_path = os.path.join(class_image_dir, image_name)
            audio_to_melspectrogram(audio_path, image_path, sr, n_mels, n_fft, hop_length)

# Split dataset into train and test
train_dir = 'path/to/train'  # Update with actual path
test_dir = 'path/to/test'  # Update with actual path
print(f"Creating train directory: {train_dir}")
print(f"Creating test directory: {test_dir}")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in classes:
    class_image_dir = os.path.join(image_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    print(f"Creating train class directory: {train_class_dir}")
    print(f"Creating test class directory: {test_class_dir}")
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    images = [f for f in os.listdir(class_image_dir) if f.endswith('.jpg')]
    print(f"Found {len(images)} images in {class_image_dir}")
    train_images, test_images = train_test_split(images, test_size=0.25, random_state=42)
    print(f"Split for {class_name}: {len(train_images)} train, {len(test_images)} test images")

    for img in train_images:
        shutil.copy(os.path.join(class_image_dir, img), os.path.join(train_class_dir, img))
        print(f"Copied to train: {img}")
    for img in test_images:
        shutil.copy(os.path.join(class_image_dir, img), os.path.join(test_class_dir, img))
        print(f"Copied to test: {img}")

# Data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
}
print("Defined data transformations")

# Load datasets
print(f"Loading train dataset from: {train_dir}")
train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
print(f"Loading test dataset from: {test_dir}")
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Train loader created with {len(train_dataset)} samples")
print(f"Test loader created with {len(test_dataset)} samples")


# Define CNN model
class BabyCryCNN(nn.Module):
    def __init__(self, num_classes):
        super(BabyCryCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()
        print("Initialized CNN model")

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = BabyCryCNN(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Model, criterion, and optimizer initialized")


# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs=20):
    model.train()
    print("Starting model training")
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"Epoch {epoch + 1}/{num_epochs} started")
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        epoch_acc = 100 * correct / total
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {epoch_acc:.2f}%')


# Evaluation function
def evaluate_model(model, test_loader, criterion):
    model.eval()
    print("Starting model evaluation")
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(f"Test Batch {batch_idx}, Loss: {loss.item():.4f}")
    test_acc = 100 * correct / total
    print(f'Test Loss: {running_loss / len(test_loader):.4f}, Test Accuracy: {test_acc:.2f}%')
    return test_acc


# Train and evaluate
print("Initiating training and evaluation")
train_model(model, train_loader, criterion, optimizer, num_epochs=20)
test_acc = evaluate_model(model, test_loader, criterion)
print(f"Final Testing Accuracy: {test_acc:.2f}%")