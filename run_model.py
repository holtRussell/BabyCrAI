import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from moviepy import VideoFileClip


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


# Utility function to convert video to audio
def video_to_audio(video_path, audio_path):
    """
    Extracts audio from a video file and saves it as a .wav file.
    """
    try:
        print(f"Converting video: {video_path} to audio: {audio_path}")
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        print("Conversion successful.")
        return True
    except Exception as e:
        print(f"Error converting video to audio: {e}")
        return False
    finally:
        if 'video_clip' in locals():
            video_clip.close()


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
def predict_cry(file_path, model_path, device):
    # Define classes (match your trained model)
    classes = ['hungry', 'burping', 'discomfort', 'belly_pain', 'tired', 'unknown']
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

    # Check file extension and convert if necessary
    temp_audio_path = None
    if file_path.lower().endswith('.mov'):
        temp_audio_path = "temp_audio.wav"
        if not video_to_audio(file_path, temp_audio_path):
            return "Error: Could not convert video to audio", 0.0, {}
        audio_path_to_process = temp_audio_path
    else:
        audio_path_to_process = file_path

    # Preprocess the audio
    temp_image_path = "temp_mel_spec.jpg"
    input_tensor = preprocess_audio(audio_path_to_process, temp_image_path, transform)
    input_tensor = input_tensor.to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    # Get the predicted class and its confidence
    predicted_class_idx = torch.argmax(probabilities).item()
    confidence = probabilities[predicted_class_idx].item()
    predicted_class = classes[predicted_class_idx]

    # Get all confidence scores
    all_confidences = {classes[i]: probabilities[i].item() for i in range(len(classes))}

    # Clean up temporary files
    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)
    if temp_audio_path and os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

    # Return result
    return predicted_class, confidence, all_confidences


# Example usage
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    audio_file = "data/baby_cry_sense/Baby Cry Dataset/hungry/hu-14.3gp"  # Example audio file
    video_file = "real_world_data/IMG_3767.mov"  # Replace with your video file path
    model_path = "DeepInfant_V2.mlmodel"  # Replace with your .pth file path

    # Example for an audio file
    class_name, confidence_score, all_scores = predict_cry(video_file, model_path, device)
    print(f"Predicted class: {class_name}")
    print(f"Confidence score for '{class_name}': {confidence_score:.2f}")
    print("\nConfidence scores for all categories:")
    for category, score in all_scores.items():
        print(f"  - {category}: {score:.2f}")
