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
import soundfile as sf  # For saving augmented audio files


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
        mel_db = mel_db.unsqueeze(1).repeat(1, 3, 1, 1)  # [batch, 3, n_mels, time_frames]
        return mel_db


# Define Hybrid CNN-LSTM model with batch norm
class BabyCryHybrid(nn.Module):
    def __init__(self, num_classes, max_length=16000, n_fft=2048, hop_length=512):
        super(BabyCryHybrid, self).__init__()
        # Mel-spectrogram layer is now part of the model
        self.mel_layer = MelSpectrogramLayer(n_fft=n_fft, hop_length=hop_length)
        self.max_length = max_length

        # Increased CNN channel sizes for better feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Calculate LSTM input size dynamically based on output of conv layers
        self.lstm_input_size = 128 * (128 // (2 ** 3)) * (
                    int(max_length / hop_length) // (2 ** 3))  # Approx time frames
        self.lstm = nn.LSTM(self.lstm_input_size, 128, num_layers=2, bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256, num_classes)
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
        batch_size, channels, height, width = x.size()
        x = x.view(batch_size, width, -1)  # [batch, time_frames, features]
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
    def __init__(self, data_dir, augment_dir=None, transform=None, max_length=16000):
        self.data_dir = Path(data_dir)
        self.augment_dir = Path(augment_dir) if augment_dir else None
        self.transform = transform
        self.max_length = max_length
        self.classes = sorted(os.listdir(data_dir))  # e.g., ['hungry', 'burping', ...]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.audio_paths = []
        self.labels = []

        # Load from original directory
        for cls_idx, cls_name in enumerate(self.classes):
            cls_path = self.data_dir / cls_name
            for audio_file in cls_path.glob('*.wav'):
                self.audio_paths.append(audio_file)
                self.labels.append(cls_idx)

        # Load from augmented directory if provided
        if self.augment_dir and self.augment_dir.exists():
            for cls_idx, cls_name in enumerate(self.classes):
                cls_aug_path = self.augment_dir / cls_name
                if cls_aug_path.exists():
                    for audio_file in cls_aug_path.glob('*.wav'):
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

# TODO -- Fix the conversion to be more like previous projects.
# Create a custom code that's more similar to what you want.

# Custom collate function to pad batches
def pad_collate_fn(batch):
    # Sort by length to minimize padding
    batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
    waveforms, labels = zip(*batch)

    # Find the longest sequence in the batch
    max_len = max(w.size(0) for w in waveforms)

    # Pad sequences
    padded_waveforms = torch.stack(
        [torch.nn.functional.pad(w, (0, max_len - w.size(0))) for w in waveforms]
    )
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_waveforms, labels


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda', patience=10):
    model = model.to(device)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
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
                torch.save(best_model_wts, 'best_baby_cry_model.pth')
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


# Evaluation function
def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = running_corrects.double() / len(test_loader.dataset)
    return epoch_acc


# Audio augmentation function
def augment_audio_dataset(source_dir, dest_dir, sr=16000):
    """
    Augments audio files in the source directory and saves them to the destination directory.
    Applies noise, time stretching, volume changes, and pitch shifting.

    Args:
        source_dir (str or Path): Directory containing subdirectories of class-wise audio files.
        dest_dir (str or Path): Directory to save augmented audio files.
        sr (int): Sample rate for audio processing (default: 16000 Hz).
    """
    os.makedirs(dest_dir, exist_ok=True)

    for class_name in os.listdir(source_dir):
        class_source_dir = Path(source_dir) / class_name
        class_dest_dir = Path(dest_dir) / class_name
        os.makedirs(class_dest_dir, exist_ok=True)

        for audio_file in class_source_dir.glob('*.wav'):
            if audio_file.is_file():
                y, _ = librosa.load(audio_file, sr=sr)
                base_name = audio_file.stem

                # Define augmentation variations
                variations = [
                    ('orig', y),  # Original audio as baseline
                    ('noise', y + np.random.normal(0, 0.01, y.shape)),  # Add Gaussian noise
                    ('stretch11', librosa.effects.time_stretch(y, rate=1.1)),  # Speed up by 10%
                    ('stretch09', librosa.effects.time_stretch(y, rate=0.9)),  # Slow down by 10%
                    ('vol12', np.clip(y * 1.2, -1.0, 1.0)),  # Increase volume by 20%
                    ('vol08', y * 0.8),  # Decrease volume by 20%
                    ('pitch_up', librosa.effects.pitch_shift(y, sr=sr, n_steps=2)),  # Pitch up by 2 semitones
                    ('pitch_down', librosa.effects.pitch_shift(y, sr=sr, n_steps=-2))  # Pitch down by 2 semitones
                ]

                # Process each variation and save as .wav
                for var_name, y_var in variations:
                    augmented_file = class_dest_dir / f"{base_name}_{var_name}.wav"
                    if not augmented_file.exists():  # Skip if already exists to avoid overwriting
                        sf.write(augmented_file, y_var, sr)
                        print(f"Saved augmented audio: {augmented_file}")
                    else:
                        print(f"Skipping existing augmented audio: {augmented_file}")


if __name__ == "__main__":
    # Define data directories
    SOURCE_DIR = Path("data/donate_a_cry/donateacry_corpus")
    AUGMENTED_DATASET_DIR = Path("data/donate_a_cry/donateacry_corpus_augmented_dataset")

    # Augment the dataset to the new directory
    if not AUGMENTED_DATASET_DIR.exists():
        augment_audio_dataset(SOURCE_DIR, AUGMENTED_DATASET_DIR)
    else:
        print(f"Augmented dataset directory {AUGMENTED_DATASET_DIR} already exists. Skipping augmentation.")

    # Load the full augmented dataset
    print("Loading full dataset")
    total_dataset = BabyCryDataset(AUGMENTED_DATASET_DIR)
    if len(total_dataset) == 0:
        raise ValueError(f"No audio files found in {AUGMENTED_DATASET_DIR}. Check directory and augmentation.")

    # Split the dataset
    print("splitting the dataset")
    train_size = int(0.7 * len(total_dataset))
    val_size = int(0.15 * len(total_dataset))
    test_size = len(total_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        total_dataset, [train_size, val_size, test_size]
    )

    # Define subdirectories within the augmented dataset directory with label folders
    train_dir = "data/audio/train"
    val_dir = "data/audio/val"
    test_dir = "data/audio/test"
    print("defining subdirectories")
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_path, exist_ok=True)
        for class_name in total_dataset.classes:
            os.makedirs(dir_path / class_name, exist_ok=True)

    # Assign datasets to their respective directories with label subdirectories
    print("Adding Datasets to directories")
    train_indices, val_indices, test_indices = train_dataset.indices, val_dataset.indices, test_dataset.indices
    for idx, (audio_path, label) in enumerate(zip(total_dataset.audio_paths, total_dataset.labels)):
        class_name = total_dataset.classes[label]
        dest_dir = train_dir / class_name if idx in train_indices else val_dir / class_name if idx in val_indices else test_dir / class_name
        dest_file = dest_dir / audio_path.name
        if not dest_file.exists():
            shutil.copy(audio_path, dest_file)
            print(f"Copied {audio_path.name} to {dest_dir}")
        else:
            print(f"File {audio_path.name} already exists in {dest_dir}, skipping.")

    # Reload datasets from split directories to ensure correct loading
    train_dataset = BabyCryDataset(train_dir)
    val_dataset = BabyCryDataset(val_dir)
    test_dataset = BabyCryDataset(test_dir)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=pad_collate_fn)
    print(f"Train loader created with {len(train_dataset)} samples")
    print(f"Val loader created with {len(val_dataset)} samples")
    print(f"Test loader created with {len(test_dataset)} samples")

    # Get class names from one of the datasets
    classes = train_dataset.classes

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model, loss, and optimizer
    model = BabyCryHybrid(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("Model, criterion, and optimizer initialized")

    # Train and evaluate
    print("Initiating training and evaluation")
    try:
        model, history = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50,
                                     device=device)
        # Save the model
        torch.save(model.state_dict(), 'baby_cry_model.pth')
        print("Model saved as baby_cry_model.pth")
        test_acc = evaluate_model(model, test_loader, criterion, device)
        print(f"Final Testing Accuracy: {test_acc * 100:.2f}%")
    except Exception as e:
        print(f"Training halted due to error: {str(e)}")
        raise