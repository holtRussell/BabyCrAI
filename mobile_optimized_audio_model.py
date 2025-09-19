import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


# Utility function to convert audio to mel-spectrogram for visualization/debugging
def audio_to_melspectrogram(y, sr, save_path, n_mels=128, n_fft=2048, hop_length=512):
    """Convert audio to mel-spectrogram image for debugging/visualization"""
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


# Optimized MelSpectrogramLayer with proper normalization
class MelSpectrogramLayer(nn.Module):
    def __init__(self, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512):
        super(MelSpectrogramLayer, self).__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            normalized=True,
            f_min=0.0,
            f_max=sample_rate / 2
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)
        # Normalization layer for consistent input to CNN
        self.norm = nn.InstanceNorm2d(n_mels)

    def forward(self, x):
        # x: [batch, samples]
        mel = self.mel_spec(x)  # [batch, n_mels, time_frames]
        mel_db = self.db_transform(mel)  # [batch, n_mels, time_frames]

        # Normalize the spectrogram
        mel_db = self.norm(mel_db)

        # Add channel dimension and repeat to simulate 3 channels for CNN
        mel_db = mel_db.unsqueeze(1)  # [batch, 1, n_mels, time_frames]
        mel_db = mel_db.repeat(1, 3, 1, 1)  # [batch, 3, n_mels, time_frames]

        return mel_db


# Optimized Hybrid CNN-LSTM model for mobile deployment
class BabyCryHybridLite(nn.Module):
    def __init__(self, num_classes, sample_rate=16000, n_mels=128, n_fft=2048, hop_length=512, max_length=16000):
        super(BabyCryHybridLite, self).__init__()
        self.max_length = max_length
        self.mel_layer = MelSpectrogramLayer(sample_rate, n_mels, n_fft, hop_length)

        # Lightweight CNN layers with depthwise separable convolutions for mobile
        # First block: 3 -> 16 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(2, 2)

        # Second block: 16 -> 32 channels
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Third block: 32 -> 64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(2, 2)

        # Calculate LSTM input size after pooling
        # Input: [n_mels, time_frames] -> after 3 pools: [n_mels/8, time_frames/8]
        self.feature_size = (n_mels // 8) * (max_length // (hop_length * 8)) * 64
        self.lstm = nn.LSTM(self.feature_size // 64, 64, num_layers=1,
                            bidirectional=False, batch_first=True, dropout=0.2)

        # Lightweight classifier
        self.fc1 = nn.Linear(64, 64)  # Reduced from 128
        self.bn_fc1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Pad or truncate audio
        x = self._pad_or_truncate(x, self.max_length)

        # Convert to mel-spectrogram
        x = self.mel_layer(x)  # [batch, 3, n_mels, time_frames]

        # CNN feature extraction
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))

        # Reshape for LSTM: [batch, time_steps, features]
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 3, 1, 2)  # [batch, width, channels, height]
        x = x.reshape(batch_size, width, -1)  # [batch, time_steps, features]

        # LSTM processing
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last time step

        # Classification
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

    def _pad_or_truncate(self, x, max_length):
        """Pad or truncate audio to fixed length"""
        batch_size = x.size(0)
        if x.size(1) < max_length:
            pad_size = max_length - x.size(1)
            x = torch.nn.functional.pad(x, (0, pad_size))
        elif x.size(1) > max_length:
            x = x[:, :max_length]
        return x


# Enhanced Custom dataset with better augmentation and balance
class AudioDataset(Dataset):
    def __init__(self, root_dir, max_length=16000, target_sr=16000, stride=8000,
                 max_snippets_per_file=3, augment=False, noise_level=0.005):
        """
        Enhanced audio dataset with on-the-fly augmentation
        """
        self.root_dir = root_dir
        self.max_length = max_length
        self.target_sr = target_sr
        self.stride = stride
        self.max_snippets_per_file = max_snippets_per_file
        self.augment = augment
        self.noise_level = noise_level

        self.classes = ['hungry', 'burping', 'discomfort', 'belly_pain', 'tired']
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []

        # Load all audio files and create snippets
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.exists(cls_dir):
                print(f"Warning: Directory {cls_dir} does not exist, skipping.")
                continue

            for f in os.listdir(cls_dir):
                if f.endswith('.wav'):
                    self._process_audio_file(cls_dir, f, cls)

        self.class_counts = Counter([label for _, label, _ in self.samples])
        print(f"Dataset initialized. Total samples: {len(self.samples)}")
        print("Class distribution:", {self.classes[k]: v for k, v in self.class_counts.items()})

    def _process_audio_file(self, cls_dir, filename, class_name):
        """Process individual audio file and create snippets"""
        path = os.path.join(cls_dir, filename)
        try:
            waveform, sample_rate = torchaudio.load(path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.target_sr
                )
                waveform = resampler(waveform)

            waveform = waveform.squeeze(0)  # Remove channel dim

            # Create snippets
            if len(waveform) <= self.max_length:
                num_snippets = 1
            else:
                num_snippets = min(
                    self.max_snippets_per_file,
                    (len(waveform) - self.max_length) // self.stride + 1
                )

            for i in range(num_snippets):
                self.samples.append((path, self.class_to_idx[class_name], i))

        except Exception as e:
            print(f"Error processing {path}: {str(e)}")

    def _augment_audio(self, waveform):
        """Apply random augmentation to audio"""
        if not self.augment:
            return waveform

        # Random augmentation selection
        aug_type = np.random.choice(['none', 'noise', 'volume', 'stretch'], p=[0.4, 0.3, 0.2, 0.1])

        if aug_type == 'noise':
            noise = torch.randn_like(waveform) * self.noise_level
            return waveform + noise
        elif aug_type == 'volume':
            vol_factor = np.random.uniform(0.8, 1.2)
            return waveform * vol_factor
        elif aug_type == 'stretch':
            # Simple time stretch simulation by resampling
            stretch_factor = np.random.uniform(0.9, 1.1)
            # This is a simplified version - for real stretching use librosa
            return waveform
        return waveform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, snippet_idx = self.samples[idx]

        try:
            waveform, sample_rate = torchaudio.load(path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sample_rate != self.target_sr:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=self.target_sr
                )
                waveform = resampler(waveform)

            waveform = waveform.squeeze(0)

            # Extract snippet
            start = snippet_idx * self.stride
            end = start + self.max_length
            snippet = waveform[start:end]

            # Pad if too short
            if len(snippet) < self.max_length:
                pad_size = self.max_length - len(snippet)
                snippet = torch.nn.functional.pad(snippet, (0, pad_size))

            # Apply augmentation during training
            if self.augment:
                snippet = self._augment_audio(snippet)

            # Normalize to [-1, 1]
            snippet = torch.clamp(snippet / (torch.max(torch.abs(snippet)) + 1e-8), -1.0, 1.0)

            return snippet, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading sample {idx}: {str(e)}")
            # Return zero tensor as fallback
            return torch.zeros(self.max_length), torch.tensor(0, dtype=torch.long)


# Enhanced training function with comprehensive logging and monitoring
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cpu'):
    """Enhanced training with detailed monitoring and early stopping"""
    early_stopper = EarlyStopper(patience=7, min_delta=0.001)
    best_val_acc = 0.0
    patience_counter = 0

    print("Starting model training")
    print(f"Training on {len(train_loader.dataset)} samples, validating on {len(val_loader.dataset)} samples")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        print(f"\nEpoch {epoch + 1}/{num_epochs} - Training")
        epoch_start = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
        if epoch_start:
            epoch_start.record()

        try:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Check for unstable loss
                if torch.isnan(loss) or loss.item() > 10:
                    print(f"Warning: Unstable loss at batch {batch_idx}: {loss.item()}")
                    torch.save(model.state_dict(), f'model_backup_epoch_{epoch}.pth')
                    break

                loss.backward()
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Detailed logging every 20 batches
                if batch_idx % 20 == 0:
                    batch_acc = 100 * correct / total
                    batch_counts = Counter(labels.cpu().numpy())
                    print(f"  Batch {batch_idx}/{len(train_loader)} - "
                          f"Loss: {loss.item():.4f}, Acc: {batch_acc:.2f}%, "
                          f"Distribution: {batch_counts}")

        except Exception as e:
            print(f"Error in training epoch {epoch + 1}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

        # Calculate epoch metrics
        epoch_acc = 100 * correct / total
        avg_loss = running_loss / len(train_loader)

        # GPU timing if available
        if device.type == 'cuda' and epoch_start:
            epoch_end = torch.cuda.Event(enable_timing=True)
            epoch_end.record()
            torch.cuda.synchronize()
            epoch_time = epoch_start.elapsed_time(epoch_end) / 1000
            print(f"  Epoch time: {epoch_time:.2f}s")

        print(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Train Acc: {epoch_acc:.2f}%")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        print(f"Epoch {epoch + 1} - Validation")
        with torch.no_grad():
            try:
                for batch_idx, (inputs, labels) in enumerate(val_loader):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

                    if batch_idx % 10 == 0:
                        print(f"  Val Batch {batch_idx}/{len(val_loader)} - Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"Error in validation epoch {epoch + 1}: {str(e)}")
                import traceback
                traceback.print_exc()
                raise

        val_acc = 100 * val_correct / val_total
        val_loss_avg = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} - Val Loss: {val_loss_avg:.4f}, Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss_avg,
                'val_acc': val_acc,
                'train_loss': avg_loss,
                'train_acc': epoch_acc
            }, 'best_baby_cry_model.pth')
            print(f"  New best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{early_stopper.patience}")

        # Early stopping
        if early_stopper.early_stop(val_loss_avg):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    return best_val_acc


# Enhanced evaluation function with confusion matrix
def evaluate_model(model, test_loader, criterion, device, class_names):
    """Enhanced evaluation with detailed metrics"""
    model.eval()
    print("\nStarting model evaluation")

    all_predictions = []
    all_labels = []
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if batch_idx % 5 == 0:
                batch_acc = 100 * (predicted == labels).sum().item() / labels.size(0)
                print(f"Test Batch {batch_idx}/{len(test_loader)} - "
                      f"Loss: {loss.item():.4f}, Batch Acc: {batch_acc:.2f}%")

    test_acc = 100 * correct / total
    avg_loss = running_loss / len(test_loader)

    # Calculate per-class accuracy
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(all_labels, all_predictions)
    print(f"\nTest Loss: {avg_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    print("\nPer-class accuracy:")
    for i, class_name in enumerate(class_names):
        class_correct = sum(cm[i][i] for i in range(len(class_names)))
        class_total = sum(cm[i])
        if class_total > 0:
            print(f"  {class_name}: {100 * class_correct / class_total:.2f}% "
                  f"({class_correct}/{class_total})")

    print(
        f"\nDetailed classification report:\n{classification_report(all_labels, all_predictions, target_names=class_names)}")

    return test_acc


# Enhanced EarlyStopper with validation accuracy tracking
class EarlyStopper:
    def __init__(self, patience=7, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.best_epoch = 0

    def early_stop(self, validation_loss):
        if validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            self.best_epoch = 0
        else:
            self.counter += 1
            self.best_epoch += 1

            if self.counter >= self.patience:
                print(f"No improvement for {self.patience} epochs. Best loss: {self.best_loss:.4f}")
                return True
        return False


# Main execution with comprehensive setup
def main():
    # Configuration
    CONFIG = {
        'sr': 16000,
        'max_length': 16000,
        'batch_size': 16,  # Reduced for mobile optimization
        'num_epochs': 50,
        'learning_rate': 0.001,
        'classes': ['hungry', 'burping', 'discomfort', 'belly_pain', 'tired']
    }

    print("=== Baby Cry Audio Classification - Mobile Optimized ===")
    print(f"Configuration: {CONFIG}")

    # Paths
    audio_dir = 'data/total_dataset/'
    augmented_dir = 'data/augmented_audio'
    train_dir = 'data/audio/train'
    val_dir = 'data/audio/val'
    test_dir = 'data/audio/test'

    print(f"Source audio directory: {audio_dir}")
    print(f"Augmented audio directory: {augmented_dir}")

    # Create directories
    os.makedirs(augmented_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Generate augmented audio files (only if not exists)
    classes = CONFIG['classes']
    sr = CONFIG['sr']

    print("\n=== Data Augmentation ===")
    for class_name in classes:
        class_audio_dir = os.path.join(audio_dir, class_name)
        class_augmented_dir = os.path.join(augmented_dir, class_name)

        print(f"\nProcessing class: {class_name}")
        os.makedirs(class_augmented_dir, exist_ok=True)

        if not os.path.exists(class_audio_dir):
            print(f"Warning: {class_audio_dir} does not exist, skipping.")
            continue

        originals = [f for f in os.listdir(class_audio_dir) if f.endswith('.wav')]
        print(f"Found {len(originals)} original files")

        for audio_file in originals:
            audio_path = os.path.join(class_audio_dir, audio_file)
            base_name = audio_file.replace('.wav', '')

            # Copy original
            orig_path = os.path.join(class_augmented_dir, audio_file)
            if not os.path.exists(orig_path):
                shutil.copy(audio_path, orig_path)
                print(f"  Copied original: {audio_file}")

            # Only create augmentations if they don't exist
            if len([f for f in os.listdir(class_augmented_dir) if f.startswith(base_name) and '_noise' in f]) == 0:
                try:
                    y, _ = librosa.load(audio_path, sr=sr)

                    # Create augmentations
                    augmentations = [
                        ('noise', y + np.random.normal(0, 0.01, y.shape)),
                        ('stretch11', librosa.effects.time_stretch(y, rate=1.1)),
                        ('stretch09', librosa.effects.time_stretch(y, rate=0.9)),
                        ('vol12', np.clip(y * 1.2, -1.0, 1.0)),
                        ('vol08', y * 0.8)
                    ]

                    for aug_name, y_aug in augmentations:
                        aug_path = os.path.join(class_augmented_dir, f"{base_name}_{aug_name}.wav")
                        if not os.path.exists(aug_path):
                            waveform_tensor = torch.from_numpy(y_aug).unsqueeze(0).float()
                            torchaudio.save(aug_path, waveform_tensor, sr)
                            print(f"  Created augmentation: {base_name}_{aug_name}")

                except Exception as e:
                    print(f"  Error augmenting {audio_file}: {str(e)}")

    # Split dataset
    print("\n=== Dataset Splitting ===")
    for class_name in classes:
        class_augmented_dir = os.path.join(augmented_dir, class_name)
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)

        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)

        all_files = [f for f in os.listdir(class_augmented_dir) if f.endswith('.wav')]
        if not all_files:
            print(f"No files found for {class_name}, skipping.")
            continue

        print(f"\n{class_name}: {len(all_files)} total files")

        # 70/15/15 split
        train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

        print(f"  Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

        # Copy files
        for f in train_files:
            shutil.copy(os.path.join(class_augmented_dir, f), os.path.join(train_class_dir, f))
        for f in val_files:
            shutil.copy(os.path.join(class_augmented_dir, f), os.path.join(val_class_dir, f))
        for f in test_files:
            shutil.copy(os.path.join(class_augmented_dir, f), os.path.join(test_class_dir, f))

    # Load datasets with augmentation for training only
    print("\n=== Loading Datasets ===")
    train_dataset = AudioDataset(
        train_dir,
        max_length=CONFIG['max_length'],
        target_sr=CONFIG['sr'],
        stride=8000,
        max_snippets_per_file=3,
        augment=True,  # Augmentation for training
        noise_level=0.005
    )

    val_dataset = AudioDataset(
        val_dir,
        max_length=CONFIG['max_length'],
        target_sr=CONFIG['sr'],
        stride=8000,
        max_snippets_per_file=3,
        augment=False  # No augmentation for validation
    )

    test_dataset = AudioDataset(
        test_dir,
        max_length=CONFIG['max_length'],
        target_sr=CONFIG['sr'],
        stride=8000,
        max_snippets_per_file=3,
        augment=False
    )

    # Create data loaders with class balancing
    print("\n=== Creating Data Loaders ===")
    num_classes = len(classes)

    # Calculate class weights for loss function
    train_class_counts = Counter([label for _, label, _ in train_dataset.samples])
    total_samples = sum(train_class_counts.values())
    class_weights = torch.tensor([
        total_samples / (num_classes * train_class_counts.get(i, 1))
        for i in range(num_classes)
    ], dtype=torch.float)

    # Normalize and clamp weights
    class_weights = class_weights / class_weights.sum() * num_classes
    class_weights = torch.clamp(class_weights, 0.5, 3.0)

    print(f"Class weights: {dict(zip(classes, class_weights.tolist()))}")

    # Weighted sampler for training
    sample_weights = [1.0 / (train_class_counts[label] + 1)
                      for _, label, _ in train_dataset.samples]
    sample_weights = torch.tensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Data loaders created:")
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} samples, {len(test_loader)} batches")

    # Initialize device and model
    print("\n=== Model Setup ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    model = BabyCryHybridLite(
        num_classes=len(classes),
        sample_rate=CONFIG['sr'],
        max_length=CONFIG['max_length']
    ).to(device)

    # Model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created: {total_params:,} total params, {trainable_params:,} trainable")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    print("Training configuration:")
    print(f"  Optimizer: AdamW (lr={CONFIG['learning_rate']}, wd=1e-4)")
    print(f"  Scheduler: ReduceLROnPlateau")
    print(f"  Loss: CrossEntropy with class weights")

    # Train model
    print("\n=== Starting Training ===")
    try:
        best_val_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer,
            num_epochs=CONFIG['num_epochs'], device=device
        )

        # Load best model for evaluation
        checkpoint = torch.load('best_baby_cry_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"\nLoaded best model from epoch {checkpoint['epoch'] + 1}")
        print(f"Best validation accuracy: {checkpoint['val_acc']:.2f}%")

        # Evaluate on test set
        print("\n=== Test Evaluation ===")
        test_acc = evaluate_model(model, test_loader, criterion, device, classes)

        print(f"\n=== Training Summary ===")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Model saved as: best_baby_cry_model.pth")

    except Exception as e:
        print(f"\nTraining failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Save whatever we have
        torch.save(model.state_dict(), f'failed_model_epoch_{CONFIG["num_epochs"]}.pth')
        raise


if __name__ == "__main__":
    main()