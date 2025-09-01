import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


# Utility function to convert audio (y) to mel-spectrogram image
def audio_to_melspectrogram(y, sr, save_path, n_mels=128, n_fft=2048, hop_length=512):
    print(f"Converting to mel-spectrogram: {save_path}")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved mel-spectrogram image: {save_path}")


# Training function
def train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=50):
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
        scheduler.step()
        epoch_acc = 100 * correct / total
        print(
            f'Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {epoch_acc:.2f}%')


# Evaluation function with per-class metrics
def evaluate_model(model, test_loader, criterion, classes, class_to_idx, device):
    model.eval()
    print("Starting model evaluation")
    correct = 0
    total = 0
    running_loss = 0.0
    all_preds = []
    all_labels = []
    class_correct = {cls: 0 for cls in classes}
    class_total = {cls: 0 for cls in classes}
    class_loss = {cls: 0.0 for cls in classes}

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Per-class metrics
            for label, pred in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                class_name = [k for k, v in class_to_idx.items() if v == label][0]
                class_total[class_name] += 1
                class_loss[class_name] += loss.item() / labels.size(0)  # Normalize loss per sample
                if label == pred:
                    class_correct[class_name] += 1

            print(f"Test Batch {batch_idx}, Loss: {loss.item():.4f}")

    # Compute overall metrics
    test_acc = 100 * correct / total
    test_loss = running_loss / len(test_loader)

    # Compute per-class metrics
    print("\nPer-Class Evaluation:")
    for cls in classes:
        if class_total[cls] > 0:
            class_acc = 100 * class_correct[cls] / class_total[cls]
            class_avg_loss = class_loss[cls] / class_total[cls]
            print(f"Accuracy for {cls}: {class_acc:.2f}%")
            print(f"Average Loss for {cls}: {class_avg_loss:.4f}")
        else:
            print(f"No test samples for {cls}")

    # Compute and display confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[class_to_idx[cls] for cls in classes])
    print("\nConfusion Matrix:")
    print("Rows: True Labels, Columns: Predicted Labels")
    print(f"{'':<15} {' '.join([f'{cls[:8]:<8}' for cls in classes])}")
    for i, cls in enumerate(classes):
        print(f"{cls[:8]:<15} {' '.join([f'{val:<8}' for val in cm[i]])}")

    print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")
    return test_acc


if __name__ == '__main__':
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
    classes = ['hungry', 'burping', 'discomfort', 'belly_pain', 'tired']
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
                    # ('noise', y + np.random.normal(0, 0.01, y.shape)),
                    # ('stretch11', librosa.effects.time_stretch(y, rate=1.1)),
                    # ('stretch09', librosa.effects.time_stretch(y, rate=0.9)),
                    # ('vol12', np.clip(y * 1.2, -1.0, 1.0)),
                    # ('vol08', y * 0.8)
                ]

                # Process each variation
                for var_name, y_var in variations:
                    image_path = os.path.join(class_image_dir, f"{base_name}_{var_name}.jpg")
                    if os.path.exists(image_path):
                        print(f"Skipping existing mel-spectrogram: {image_path}")
                        continue
                    audio_to_melspectrogram(y_var, sr, image_path, n_mels, n_fft, hop_length)

    # Split dataset into train and test
    train_dir = 'data/train'  # Updated path
    test_dir = 'data/test'  # Updated path
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

    # Data transformations with additional image augmentation for training
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    print("Defined data transformations")

    # Load datasets
    print(f"Loading train dataset from: {train_dir}")
    train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
    print(f"Loading test dataset from: {test_dir}")
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    print(f"Train loader created with {len(train_dataset)} samples")
    print(f"Test loader created with {len(test_dataset)} samples")

    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Use pre-trained ResNet50
    model = models.resnet50(pretrained=True)
    # Freeze all layers except the last block and classifier
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    model = model.to(device)
    print("Model initialized")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    print("Criterion, optimizer, and scheduler initialized")

    # Train and evaluate
    print("Initiating training and evaluation")
    train_model(model, train_loader, criterion, optimizer, scheduler, device, num_epochs=50)
    test_acc = evaluate_model(model, test_loader, criterion, classes, test_dataset.class_to_idx, device)
    print(f"Final Testing Accuracy: {test_acc:.2f}%")