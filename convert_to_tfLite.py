import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from mobile_optimized import BabyCryHybridLite  # Import your lightweight model class

# Define the model class (ensure it matches your trained architecture)
class BabyCryHybridLite(nn.Module):
    def __init__(self, num_classes):
        super(BabyCryHybridLite, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # Reduced from 32
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)  # Reduced from 64
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Reduced from 128
        self.bn3 = nn.BatchNorm2d(64)
        self.lstm = nn.LSTM(64 * 16, 64, num_layers=1, bidirectional=False, batch_first=True)  # Simplified LSTM
        self.fc1 = nn.Linear(64, 128)  # Reduced from 256, 512
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)
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

# Load the pre-trained model weights
num_classes = 6  # Adjust based on your model (including 'unknown')
model = BabyCryHybridLite(num_classes=num_classes)
model.load_state_dict(torch.load('baby_cry_model_lite.pth'))  # Load your pre-existing .pth file
model.eval()

# Prepare a dummy input for tracing (matches input shape: [batch_size, channels, height, width])
dummy_input = torch.randn(1, 3, 128, 128)  # Example input shape from your model

# Trace the model with TorchScript
try:
    scripted_model = torch.jit.trace(model, dummy_input)
    optimized_traced_model = optimize_for_mobile(scripted_model)
    print("Model successfully traced with TorchScript")
except Exception as e:
    print(f"TorchScript tracing failed: {e}")
    # If tracing fails (e.g., due to LSTM), use scripting instead: scripted_model = torch.jit.script(model)
    raise

# Save the scripted model as .pt asset
optimized_traced_model._save_for_lite_interpreter("baby_crAI.pt")
print("TorchScript model saved as baby_cry_model_lite_torchscript.pt")

# Optional: Test inference with TorchScript model
test_input = torch.randn(1, 3, 128, 128)
output = scripted_model(test_input)
print(f"Test output shape: {output.shape}")

