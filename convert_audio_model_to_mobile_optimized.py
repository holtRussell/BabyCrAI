import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile
import torchaudio

# Import your audio model with embedded mel-spectrogram layer
from audio_embedded_model import BabyCryHybridLite  # Import your full audio model

# Load the pre-trained model weights
num_classes = 5  # Updated for 5 classes (excluding 'unknown')
model = BabyCryHybridLite(num_classes=num_classes, sample_rate=16000, n_mels=128,
                         n_fft=2048, hop_length=512, max_length=16000)
model.load_state_dict(torch.load('baby_cry_model_lite.pth', map_location='cpu'))  # Load your .pth file
model.eval()
print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")

# Prepare a dummy audio input for tracing (matches your model's expected input: [batch, samples])
dummy_audio = torch.randn(1, 16000)  # Shape: [1, 16000] - 1 second of mono audio at 16kHz
print(f"Dummy audio input shape: {dummy_audio.shape}")

# Test the model with dummy input to verify it works
try:
    with torch.no_grad():
        test_output = model(dummy_audio)
        print(f"Test output shape: {test_output.shape} (expected: [{num_classes}, 5])")
        print(f"Sample output: {test_output}")
except Exception as e:
    print(f"Model forward pass failed: {e}")
    raise

# Trace the model with TorchScript
try:
    # Trace with audio input (this captures the mel-spectrogram transformation)
    traced_model = torch.jit.trace(model, dummy_audio)
    # Optimize for mobile (quantization, fusion, etc.)
    optimized_model = optimize_for_mobile(traced_model)
    print("Model successfully traced and optimized for mobile")
except Exception as e:
    print(f"TorchScript tracing failed: {e}")
    print("This is common with audio transforms. Trying scripting instead...")
    # Fallback: Use scripting if tracing fails (less optimized but more reliable)
    try:
        scripted_model = torch.jit.script(model)
        optimized_model = optimize_for_mobile(scripted_model)
        print("Model successfully scripted and optimized for mobile")
    except Exception as e2:
        print(f"Scripting also failed: {e2}")
        raise

# Save the optimized model for mobile deployment
model_filename = "baby_crAI_audio_mobile.pt"
optimized_model._save_for_lite_interpreter(model_filename)
print(f"Optimized TorchScript model saved as {model_filename}")

# Optional: Test inference with the traced/scripted model
try:
    with torch.no_grad():
        traced_output = optimized_model(dummy_audio)
        print(f"Traced model output shape: {traced_output.shape}")
        print(f"Traced model sample output: {traced_output}")
        print(f"Original vs traced output diff: {torch.max(torch.abs(test_output - traced_output))}")
except Exception as e:
    print(f"Traced model test failed: {e}")

# Save model metadata for mobile app
metadata = {
    "num_classes": num_classes,
    "classes": ['hungry', 'burping', 'discomfort', 'belly_pain', 'tired'],
    "sample_rate": 16000,
    "max_length": 16000,
    "stride": 8000,
    "input_shape": [1, 16000],
    "output_shape": [1, num_classes]
}

import json
with open("model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("Model metadata saved to model_metadata.json")

print("\n=== Conversion Complete ===")
print(f"Model file: {model_filename}")
print(f"Classes: {metadata['classes']}")
print(f"Input: {metadata['input_shape']} (1s mono audio at 16kHz)")
print(f"Output: {metadata['output_shape']} (class probabilities)")
print("\nFor mobile app inference:")
print("1. Load .pt file with PyTorch Mobile or convert to TensorFlow Lite")
print("2. Preprocess audio: mono, 16kHz, 16000 samples (pad/truncate)")
print("3. Run inference: output = model(input_tensor)")
print("4. Get prediction: class = classes[torch.argmax(output)]")