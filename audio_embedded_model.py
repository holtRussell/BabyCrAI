import torchaudio
import librosa
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


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
        mel = self.mel_spec(x)  # [batch, n_mels, time_frames]
        mel_db = self.db_transform(mel)  # [batch, n_mels, time_frames]
        # Add channel dimension and repeat to simulate 3 channels (RGB-like) for CNN input
        mel_db = mel_db.unsqueeze(1)  # [batch, 1, n_mels, time_frames]
        mel_db = mel_db.repeat(1, 3, 1, 1)  # [batch, 3, n_mels, time_frames]
        return mel_db

# Define Hybrid CNN-LSTM model with embedded mel-spectrogram layer
class BabyCryHybridLite(nn.Module):
    def __init__(self, num_classes, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512, max_length=16000):
        super(BabyCryHybridLite, self).__init__()
        self.mel_layer = MelSpectrogramLayer(sample_rate, n_mels, n_fft, hop_length)
        self.max_length = max_length  # Max audio length in samples
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Expects [batch, 3, n_mels, time_frames]
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # LSTM input size based on debug output: [batch, n_mels/8, 64 * (time_frames/8)]
        # From debug: time_frames=32, time_frames/8=4, so 64 * 4 = 256
        self.lstm_input_size = 256  # Feature size per time step: 64 * (time_frames/8)
        self.lstm = nn.LSTM(self.lstm_input_size, 64, num_layers=1, bidirectional=False, batch_first=True)
        self.fc1 = nn.Linear(64, 128)
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
        # Reshape for LSTM: [batch, n_mels/8, 64 * (time_frames/8)]
        x = x.view(x.size(0), x.size(2), -1)  # [batch, n_mels/8, 64 * (time_frames/8)]
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

# Custom dataset for raw audio files
class AudioDataset(Dataset):
    def __init__(self, root_dir, max_length=16000, target_sr=16000):
        self.root_dir = root_dir
        self.max_length = max_length
        self.target_sr = target_sr
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.endswith('.wav'):
                    self.samples.append((os.path.join(cls_dir, f), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(path)  # Load without specifying sr
        # Ensure mono by averaging channels if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        else:
            waveform = waveform  # Already mono
        # Resample to target_sr if necessary
        if sample_rate != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=self.target_sr)
            waveform = resampler(waveform)
        waveform = waveform.squeeze(0)  # Remove channel dimension after processing
        # Pad or truncate to max_length
        if len(waveform) < self.max_length:
            pad_size = self.max_length - len(waveform)
            waveform = torch.nn.functional.pad(waveform, (0, pad_size))
        else:
            waveform = waveform[:self.max_length]
        return waveform, torch.tensor(label, dtype=torch.long)

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
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {running_loss / len(train_loader):.4f}, Train Accuracy: {epoch_acc:.2f}%')

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
audio_dir = 'data/total_dataset/'  # Original audio files
augmented_dir = 'data/augmented_audio'  # New directory for augmented audio files
print(f"Source audio directory: {audio_dir}")
print(f"Augmented audio directory: {augmented_dir}")

# Parameters
sr = 16000
classes = ['hungry', 'burping', 'discomfort', 'belly_pain', 'tired', 'unknown']
print(f"Classes defined: {classes}")

# Generate augmented audio files and store them in a separate augmented directory
os.makedirs(augmented_dir, exist_ok=True)
for class_name in classes:
    class_audio_dir = os.path.join(audio_dir, class_name)
    class_augmented_dir = os.path.join(augmented_dir, class_name)
    print(f"Processing augmentations for class: {class_name}")
    print(f"Creating augmented class directory: {class_augmented_dir}")
    os.makedirs(class_augmented_dir, exist_ok=True)

    if not os.path.exists(class_audio_dir):
        print(f"Directory {class_audio_dir} does not exist, skipping.")
        continue
    originals = [f for f in os.listdir(class_audio_dir) if f.endswith('.wav')]
    for audio_file in originals:
        audio_path = os.path.join(class_audio_dir, audio_file)
        base_name = audio_file.replace('.wav', '')
        y, _ = librosa.load(audio_path, sr=sr)

        # Save original audio to augmented directory (to include it in dataset)
        orig_path = os.path.join(class_augmented_dir, audio_file)
        if not os.path.exists(orig_path):
            shutil.copy(audio_path, orig_path)
            print(f"Copied original audio to augmented directory: {orig_path}")

        # Define variations
        variations = [
            ('noise', y + np.random.normal(0, 0.01, y.shape)),
            ('stretch11', librosa.effects.time_stretch(y, rate=1.1)),
            ('stretch09', librosa.effects.time_stretch(y, rate=0.9)),
            ('vol12', np.clip(y * 1.2, -1.0, 1.0)),
            ('vol08', y * 0.8)
        ]

        # Process each variation
        for var_name, y_var in variations:
            var_path = os.path.join(class_augmented_dir, f"{base_name}_{var_name}.wav")
            if os.path.exists(var_path):
                print(f"Skipping existing augmented audio: {var_path}")
                continue
            waveform_tensor = torch.from_numpy(y_var).unsqueeze(0).float()
            torchaudio.save(var_path, waveform_tensor, sr)
            print(f"Saved augmented audio: {var_path}")

# Split dataset into train, val, and test (70/15/15) using augmented directory
train_dir = 'data/audio/train'  # Updated path
val_dir = 'data/audio/val'  # New val path
test_dir = 'data/audio/test'  # Updated path
print(f"Creating train directory: {train_dir}")
print(f"Creating val directory: {val_dir}")
print(f"Creating test directory: {test_dir}")
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

for class_name in classes:
    class_augmented_dir = os.path.join(augmented_dir, class_name)
    train_class_dir = os.path.join(train_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)
    test_class_dir = os.path.join(test_dir, class_name)
    print(f"Creating train class directory: {train_class_dir}")
    print(f"Creating val class directory: {val_class_dir}")
    print(f"Creating test class directory: {test_class_dir}")
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    all_files = [f for f in os.listdir(class_augmented_dir) if f.endswith('.wav')]
    print(f"Found {len(all_files)} audio files in {class_augmented_dir}")

    # Split into train and temp (70/30, where temp is val+test)
    train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
    # Split temp into val and test (50/50 of temp, i.e., 15/15 overall)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    print(
        f"Split for {class_name}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test files")

    for f in train_files:
        shutil.copy(os.path.join(class_augmented_dir, f), os.path.join(train_class_dir, f))
        print(f"Copied to train: {f}")
    for f in val_files:
        shutil.copy(os.path.join(class_augmented_dir, f), os.path.join(val_class_dir, f))
        print(f"Copied to val: {f}")
    for f in test_files:
        shutil.copy(os.path.join(class_augmented_dir, f), os.path.join(test_class_dir, f))
        print(f"Copied to test: {f}")

# Load datasets using custom AudioDataset
print(f"Loading train dataset from: {train_dir}")
train_dataset = AudioDataset(train_dir, max_length=16000, target_sr=16000)
print(f"Loading val dataset from: {val_dir}")
val_dataset = AudioDataset(val_dir, max_length=16000, target_sr=16000)
print(f"Loading test dataset from: {test_dir}")
test_dataset = AudioDataset(test_dir, max_length=16000, target_sr=16000)

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
model = BabyCryHybridLite(num_classes=len(classes)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
print("Model, criterion, and optimizer initialized")

# Train and evaluate
print("Initiating training and evaluation")
try:
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device=device)
    # Save the model
    torch.save(model.state_dict(), 'baby_cry_audio_embedded.pth')
    print("Model saved as baby_cry_model_lite.pth")
    test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Final Testing Accuracy: {test_acc:.2f}%")
except Exception as e:
    print(f"Training halted due to error: {str(e)}")
    raise