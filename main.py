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

# Utility function to convert audio (y) to mel-spectrogram image with log normalization
def audio_to_melspectrogram(y, sr, save_path, n_mels=128, n_fft=2048, hop_length=512):
    print(f"Converting to mel-spectrogram: {save_path}")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = librosa.amplitude_to_db(np.abs(S_dB))  # Additional log normalization

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved mel-spectrogram image: {save_path}")


# Define Hybrid CNN-LSTM model with batch norm
class BabyCryHybrid(nn.Module):
    def __init__(self, num_classes):
        super(BabyCryHybrid, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.lstm = nn.LSTM(128 * 16, 128, bidirectional=True, batch_first=True, dropout=0.3)  # Added LSTM dropout
        self.fc1 = nn.Linear(256, 512)  # Bidirectional LSTM output is 2*128
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # Reshape for LSTM: (batch, time, features) - assuming height is reduced to 1, width is time (16), channels*height=128*1=128
        x = x.view(x.size(0), x.size(3), -1)  # (batch, time=16, features=128)
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last time step output
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# Training function with early stopping
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    early_stopper = EarlyStopper(patience=5, min_delta=0)  # Use the provided class
    print("Starting model training")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        print(f"Epoch {epoch + 1}/{num_epochs} started (Training)")
        try:
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
        except Exception as e:
            print(f"Error in training epoch {epoch + 1}: {str(e)}")
            raise  # Re-raise to debug the error

        epoch_acc = 100 * correct / total
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {epoch_acc:.2f}%')

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            try:
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            except Exception as e:
                print(f"Error in validation epoch {epoch + 1}: {str(e)}")
                raise  # Re-raise to debug the error
        val_acc = 100 * val_correct / val_total
        val_loss /= len(val_loader)
        print(f'Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.2f}%')

        # Early stopping check
        if early_stopper.early_stop(val_loss):
            print(f'Early stopping at epoch {epoch + 1}')
            break

# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
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

# EarlyStopper class
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

# Paths
audio_dir = 'data/donate_a_cry/donateacry_corpus'  # Update with actual path if needed
image_dir = 'data/donate_a_cry/melspectrogram_images'  # Update with actual path if needed
print(f"Creating image directory: {image_dir}")
os.makedirs(image_dir, exist_ok=True)

# Parameters
sr = 16000
n_mels = 128
n_fft = 2048
hop_length = 512
classes = ['hungry', 'burping', 'discomfort', 'belly_pain', 'tired', 'unknown',]
print(f"Classes defined: {classes}")

# Convert audio files to mel-spectrogram images with variations, skip if already exists
for class_name in classes:
    class_audio_dir = os.path.join(audio_dir, class_name)
    class_image_dir = os.path.join(image_dir, class_name)
    print(f"Creating class image directory: {class_image_dir}")
    os.makedirs(class_image_dir, exist_ok=True)

    for audio_file in os.listdir(class_audio_dir):
        if audio_file.endswith('.wav'):
            audio_path = os.path.join(class_audio_dir, audio_file)
            base_name = audio_file.replace('.wav', '')
            y, sr_load = librosa.load(audio_path, sr=sr)

            # Define variations
            variations = [
                ('orig', y),
                ('noise', y + np.random.normal(0, 0.01, y.shape)),
                ('stretch11', librosa.effects.time_stretch(y, rate=1.1)),
                ('stretch09', librosa.effects.time_stretch(y, rate=0.9)),
                ('vol12', np.clip(y * 1.2, -1.0, 1.0)),
                ('vol08', y * 0.8)
            ]

            # Process each variation
            for var_name, y_var in variations:
                image_path = os.path.join(class_image_dir, f"{base_name}_{var_name}.jpg")
                if os.path.exists(image_path):
                    print(f"Skipping existing mel-spectrogram: {image_path}")
                    continue
                audio_to_melspectrogram(y_var, sr, image_path, n_mels, n_fft, hop_length)

# Split dataset into train, val, and test (70/15/15)
train_dir = 'data/train'  # Updated path
val_dir = 'data/val'      # New val path
test_dir = 'data/test'    # Updated path
print(f"Creating train directory: {train_dir}")
print(f"Creating val directory: {val_dir}")
print(f"Creating test directory: {test_dir}")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in classes:
    class_image_dir = os.path.join(image_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    print(f"Creating train class directory: {train_class_dir}")
    print(f"Creating val class directory: {val_class_dir}")
    print(f"Creating test class directory: {test_class_dir}")
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    images = [f for f in os.listdir(class_image_dir) if f.endswith('.jpg')]
    print(f"Found {len(images)} images in {class_image_dir}")

    # Split into train and temp (85/15, where temp is val+test)
    train_images, temp_images = train_test_split(images, test_size=0.3, random_state=42)
    # Split temp into val and test (50/50 of temp, i.e., 15/15 overall)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

    print(
        f"Split for {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test images")

    for img in train_images:
        shutil.copy(os.path.join(class_image_dir, img), os.path.join(train_class_dir, img))
        print(f"Copied to train: {img}")
    for img in val_images:
        shutil.copy(os.path.join(class_image_dir, img), os.path.join(val_class_dir, img))
        print(f"Copied to val: {img}")
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
    'val': transforms.Compose([
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
print(f"Loading val dataset from: {val_dir}")
val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms['val'])
print(f"Loading test dataset from: {test_dir}")
test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
print(f"Train loader created with {len(train_dataset)} samples")
print(f"Val loader created with {len(val_dataset)} samples")
print(f"Test loader created with {len(test_dataset)} samples")

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model, loss, and optimizer
# Optional: Disable CuDNN if previous issue persists
# torch.backends.cudnn.enabled = False
model = BabyCryHybrid(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Model, criterion, and optimizer initialized")

# Train and evaluate
print("Initiating training and evaluation")
try:
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    # Save the model
    torch.save(model.state_dict(), 'baby_cry_model.pth')
    print("Model saved as baby_cry_model.pth")
    test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Testing Accuracy: {test_acc:.2f}%")
except Exception as e:
    print(f"Training halted due to error: {str(e)}")
    raise