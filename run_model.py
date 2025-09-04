import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt  # Added for audio_to_melspectrogram

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
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), x.size(3), -1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Utility function
def audio_to_melspectrogram(y, sr, save_path, n_mels=128, n_fft=2048, hop_length=512):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
    S_dB = librosa.power_to_db(S, ref=np.max)
    S_dB = librosa.amplitude_to_db(np.abs(S_dB))

    plt.figure(figsize=(4, 4))
    librosa.display.specshow(S_dB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Load and preprocess function
def preprocess_audio(audio_path, save_path, transform):
    y, sr = librosa.load(audio_path, sr=16000)
    audio_to_melspectrogram(y, sr, save_path)
    image = Image.open(save_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Main inference function
def predict_cry(audio_path, model_path, device):
    # Define classes (match your trained model)
    classes = ['hungry', 'burping', 'discomfort', 'belly_pain', 'tired', 'unknown', ]
    num_classes = len(classes)

    # Initialize model
    model = BabyCryHybrid(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Preprocess the audio
    temp_image_path = "temp_mel_spec.jpg"
    input_tensor = preprocess_audio(audio_path, temp_image_path, transform)
    input_tensor = input_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class_idx].item()

    # Clean up temporary file
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    # Return result
    predicted_class = classes[predicted_class_idx]
    return predicted_class, confidence

# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    audio_file = "data/baby_cry_sense/Baby Cry Dataset/hungry/hu-14.3gp"  # Replace with your audio file path
    model_path = "baby_cry_model_81%_accuracy.pth"  # Replace with your .pth file path
    class_name, confidence_score = predict_cry(audio_file, model_path, device)
    print(f"Predicted class: {class_name}, Confidence: {confidence_score:.2f}")