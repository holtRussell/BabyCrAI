import torch
import torch.nn as nn
import torchaudio
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchaudio
import numpy as np
from pathlib import Path
import os
from sklearn.model_selection import train_test_split

# Define mel-spectrogram transformation as part of the model
class MelSpectrogramLayer(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512):
        super(MelSpectrogramLayer, self).__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            normalized=True
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def forward(self, x):
        # x: [batch, samples]
        mel = self.mel_spec(x)
        mel_db = self.db_transform(mel)  # [batch, n_mels, time_frames]
        # Ensure 3 channels by repeating (simulating RGB for CNN input)
        mel_db = mel_db.repeat(1, 3, 1, 1)  # [batch, 3, n_mels, time_frames]
        return mel_db

# Define Hybrid CNN-LSTM model with embedded mel-spectrogram layer
class BabyCryHybridLite(nn.Module):
    def __init__(self, num_classes, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512, max_length=16000):
        super(BabyCryHybridLite, self).__init__()
        self.mel_layer = MelSpectrogramLayer(sample_rate, n_mels, n_fft, hop_length)
        self.max_length = max_length  # Max audio length in samples
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Reduced from 32
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced from 64
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Reduced from 128
        self.bn3 = nn.BatchNorm2d(64)
        # LSTM input size based on n_mels * time_frames after pooling
        self.lstm = nn.LSTM(64 * 16, 64, num_layers=1, bidirectional=False, batch_first=True)  # Simplified LSTM
        self.fc1 = nn.Linear(64, 128)  # Reduced from 256, 512
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [batch, samples]
        # Pad or truncate audio to max_length
        x = self._pad_or_truncate(x, self.max_length)
        # Convert to mel-spectrogram
        x = self.mel_layer(x)  # [batch, 3, n_mels, time_frames]
        # CNN processing
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        # Reshape for LSTM: [batch, time_frames, features]
        x = x.view(x.size(0), x.size(2), -1)  # [batch, time_frames, 64 * 16]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take the last time step
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def _pad_or_truncate(self, x, max_length):
        # x: [batch, samples]
        if x.size(1) < max_length:
            pad_size = max_length - x.size(1)
            x = torch.nn.functional.pad(x, (0, pad_size))  # Pad with zeros at the end
        elif x.size(1) > max_length:
            x = x[:, :max_length]  # Truncate
        return x


# Custom Dataset for WAV files
class BabyCryDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_length=16000):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.max_length = max_length
        self.classes = sorted(os.listdir(data_dir))  # e.g., ['hungry', 'burping', ...]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.audio_paths = []
        self.labels = []

        for cls_idx, cls_name in enumerate(self.classes):
            cls_path = self.data_dir / cls_name
            for audio_file in cls_path.glob('*.wav'):
                self.audio_paths.append(audio_file)
                self.labels.append(cls_idx)

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        if waveform.size(0) > 1:  # Convert stereo to mono if needed
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)  # [samples]

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

# Transform function for audio (padding/truncation handled in model)
def audio_transform(waveform):
    return waveform

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda', patience=10):
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 60)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            data_loader = train_loader if phase == 'train' else val_loader
            for inputs, labels in data_loader:
                inputs = inputs.to(device).float()  # [batch, samples]
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(data_loader.dataset)
            epoch_acc = running_corrects.double() / len(data_loader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                torch.save(best_model_wts, 'best_baby_cry_model_lite.pth')
                patience_counter = 0
            elif phase == 'val':
                patience_counter += 1

            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # Early stopping
            if patience_counter >= patience:
                print(f'Early stopping triggered after {patience} epochs with no improvement')
                model.load_state_dict(best_model_wts)
                return model, history

    print(f'Training complete. Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history

# Example usage (adjust paths and parameters)
if __name__ == '__main__':
    # Define data directory
    data_dir = Path('data/donate_a_cry/donateacry_corpus')

    # Create datasets
    train_dataset = BabyCryDataset(data_dir, transform=audio_transform)
    # Split dataset (assuming no pre-split val/test for simplicity)
    train_size = int(0.7 * len(train_dataset))
    val_size = int(0.15 * len(train_dataset))
    test_size = len(train_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size, test_size]
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BabyCryHybridLite(num_classes=6, max_length=16000).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)

    # Save final model
    torch.save(model.state_dict(), 'baby_cry_model_lite.pth')