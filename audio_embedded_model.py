import os
import shutil
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt


# Define the model class
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
        self.lstm = nn.LSTM(128 * 16, 128, bidirectional=True, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), x.size(3), -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


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
        # Correctly add a channel dimension and repeat it 3 times
        mel_db = mel_db.unsqueeze(1).repeat(1, 3, 1, 1)
        return mel_db


# Custom Dataset for WAV files
class AudioDataset(Dataset):
    def __init__(self, audio_paths, labels, transform=None, max_length=16000):
        self.audio_paths = audio_paths
        self.labels = labels
        self.transform = transform
        self.max_length = max_length

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]

        # Load audio
        waveform, _ = torchaudio.load(audio_path)

        # Pad or truncate the waveform to a fixed length
        if waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            padding = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))

        # Apply the transform (Mel-spectrogram conversion)
        if self.transform:
            mel_spec = self.transform(waveform)

        return mel_spec.squeeze(0), label


# Train model function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch + 1}/{num_epochs} - Training Loss: {epoch_loss:.4f}")

    print("Training complete.")


def augment_and_save_audio(input_base_dir, output_base_dir, sample_rate=16000):
    """
    Reads audio files from a base directory, applies augmentations, and saves them
    to a new directory with the same folder structure.

    Args:
        input_base_dir (str): The base directory containing class subfolders of WAV files.
        output_base_dir (str): The base directory where augmented audio will be saved.
        sample_rate (int): The target sample rate for audio processing.
    """
    input_base_path = Path(input_base_dir)
    output_base_path = Path(output_base_dir)

    # If the output directory exists, remove it and create a fresh one to avoid duplicates.
    if output_base_path.exists() and output_base_path.is_dir():
        print(f"Removing existing directory: {output_base_path}")
        shutil.rmtree(output_base_path)
    output_base_path.mkdir(parents=True, exist_ok=True)

    # Iterate through each class folder in the input directory
    for class_name in os.listdir(input_base_path):
        class_audio_dir = input_base_path / class_name
        if not class_audio_dir.is_dir():
            continue

        # Create the corresponding class directory in the output folder
        class_output_dir = output_base_path / class_name
        class_output_dir.mkdir(exist_ok=True)

        print(f"Processing audio files from: {class_audio_dir}")

        # Process each audio file
        for audio_file in os.listdir(class_audio_dir):
            if not audio_file.endswith('.wav'):
                continue

            audio_path = class_audio_dir / audio_file
            base_name = os.path.splitext(audio_file)[0]

            try:
                # Load audio
                y, sr_load = torchaudio.load(audio_path)
                # Ensure it's a 1D float32 numpy array, which is a reliable format for soundfile
                y = y.squeeze().numpy().astype(np.float32)
            except Exception as e:
                print(f"Error loading {audio_path}: {e}")
                continue

            # Define variations
            variations = [
                ('orig', y),
                ('noise', y + np.random.normal(0, 0.01, y.shape)),
                ('stretch11', librosa.effects.time_stretch(y, rate=1.1)),
                ('stretch09', librosa.effects.time_stretch(y, rate=0.9)),
                ('vol12', np.clip(y * 1.2, -1.0, 1.0)),
                ('vol08', y * 0.8)
            ]

            # Process each variation and save as a new audio file
            for var_name, y_var in variations:
                # Construct the output filename
                output_file_name = f"{base_name}_{var_name}.wav"
                output_path = class_output_dir / output_file_name

                # The key fix: ensure the augmented data is always float32 before saving.
                y_var_safe = y_var.astype(np.float32)

                try:
                    # Check if the augmented data is empty or malformed
                    if not y_var_safe.size > 0:
                        print(f"  - WARNING: Augmented data for '{output_file_name}' is empty. Skipping save.")
                        continue

                    # Attempt to save the file with robust error handling
                    sf.write(output_path, y_var_safe, sr_load)
                    print(f"  - Saved {output_file_name}")
                except sf.LibsndfileError as e:
                    print(f"  - ERROR: Could not save '{output_file_name}'. Issue with data format. Skipping file.")
                    print(f"    Details: {e}")
                except Exception as e:
                    print(f"  - An unexpected error occurred while saving '{output_file_name}': {e}")


# Main execution block
if __name__ == '__main__':
    # 1. Define directories
    input_directory = Path("data/donate_a_cry/donateacry_corpus")
    output_directory = Path("data/converted_audio")

    # 2. Augment and save the audio files
    augment_and_save_audio(input_directory, output_directory)

    # 3. Prepare data for training
    all_audio_paths = []
    all_labels = []
    class_to_idx = {}

    classes = sorted(os.listdir(output_directory))
    for cls_name in classes:
        cls_path = output_directory / cls_name
        if not cls_path.is_dir():
            continue

        if cls_name not in class_to_idx:
            class_to_idx[cls_name] = len(class_to_idx)

        for audio_file in cls_path.glob('*.wav'):
            all_audio_paths.append(audio_file)
            all_labels.append(class_to_idx[cls_name])

    # 4. Split data into train, validation, and test sets
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        all_audio_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    print(f"Training set size: {len(train_paths)}")
    print(f"Validation set size: {len(val_paths)}")
    print(f"Test set size: {len(test_paths)}")

    # 5. Create datasets and data loaders
    audio_transform = MelSpectrogramLayer()
    train_dataset = AudioDataset(train_paths, train_labels, transform=audio_transform)
    val_dataset = AudioDataset(val_paths, val_labels, transform=audio_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 6. Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BabyCryHybrid(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 7. Train and evaluate
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    torch.save(model.state_dict(), 'baby_cry_model_lite.pth')

    print("Training complete. Model saved.")
