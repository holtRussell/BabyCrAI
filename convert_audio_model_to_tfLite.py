import torch
import torch.onnx
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

from audio_embedded_model import BabyCryHybridLite  # Import your full audio model

# Load the pre-trained model
num_classes = 5  # Your 5 cry classes
model = BabyCryHybridLite(num_classes=num_classes, sample_rate=16000, n_mels=128,
                         n_fft=2048, hop_length=512, max_length=16000)
model.load_state_dict(torch.load('baby_cry_model_lite.pth', map_location='cpu'))
model.eval()

# Dummy audio input for export (1s mono at 16kHz)
dummy_audio = torch.randn(1, 16000)
print(f"Dummy audio input shape: {dummy_audio.shape}")

# Test forward pass
with torch.no_grad():
    test_output = model(dummy_audio)
    print(f"Test output shape: {test_output.shape}")

# Export to ONNX
torch.onnx.export(model, dummy_audio, "baby_cry_model.onnx",
                  export_params=True, opset_version=11,
                  do_constant_folding=True,
                  input_names=['audio_input'], output_names=['class_probs'],
                  dynamic_axes={'audio_input': {0: 'batch_size'}})

print("ONNX model exported successfully")

# Convert ONNX to TensorFlow
onnx_model = onnx.load("baby_cry_model.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("baby_cry_tf_saved_model")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("baby_cry_tf_saved_model")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Quantization for mobile
tflite_model = converter.convert()

with open('baby_crAI_audio.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite model saved as baby_crAI_audio.tflite")