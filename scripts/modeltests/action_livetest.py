import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- Residual Block Definition --------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)

# -------------------- Improved 3D CNN with Residual Blocks --------------------
class Improved3DCNN(nn.Module):
    def __init__(self, num_classes=101):
        super(Improved3DCNN, self).__init__()
        # Block 1
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        # Block 2
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Block 3
        self.res3 = ResidualBlock3D(128, 256, stride=2)
        
        # Block 4
        self.res4_1 = ResidualBlock3D(256, 512, stride=2)
        self.res4_2 = ResidualBlock3D(512, 512, stride=1)
        
        self.pool4 = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.res3(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)

# -------------------- Extract frames with mean–std normalization --------------------
def extract_frames_from_video(video_path, num_frames=16, img_size=(112, 112)):
    """
    Reads a video, uniformly samples `num_frames`, resizes to `img_size`,
    applies the same normalization used in training, and returns a (C, T, H, W) tensor.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = []
    frame_idx = 0
    ret = True

    while ret and frame_idx < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in indices:
            frame = cv2.resize(frame, img_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0

            # Apply the same mean-std normalization used during training
            mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
            std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
            frame = (frame - mean) / std

            frames.append(frame)
        frame_idx += 1

    cap.release()

    # If not enough frames, pad with the last frame
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1])

    if len(frames) < num_frames:
        return None

    # (T, H, W, C) -> (C, T, H, W)
    video_np = np.array(frames)
    video_np = np.transpose(video_np, (3, 0, 1, 2))
    return torch.tensor(video_np, dtype=torch.float32)

# -------------------- Device Setup --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------- Load Model --------------------
model = Improved3DCNN(num_classes=101).to(device)

# Path to your newly trained checkpoint (e.g., 50 epochs, 9000 samples)
checkpoint_path = os.path.join("..", "..", "models", "action_recognition_model_9000_50.pth")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("Loaded model from", checkpoint_path)

# -------------------- Load Class Mapping --------------------
current_dir = os.path.dirname(os.path.abspath(__file__))
class_to_idx_path = os.path.join(current_dir, "..", "..", "training", "action", "preprocess", "class_to_idx.json")
with open(class_to_idx_path, "r") as f:
    class_to_idx = json.load(f)

idx_to_class = {v: k for k, v in class_to_idx.items()}
print("Class mapping loaded.")

# -------------------- Single Video Inference --------------------
video_path = os.path.join("..", "..", "input", "lipstick.mp4")
video_tensor = extract_frames_from_video(video_path, num_frames=16, img_size=(112, 112))
if video_tensor is None:
    print(f"Failed to process video snippet at {video_path}")
else:
    video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, 3, T, H, W)
    with torch.no_grad():
        outputs = model(video_tensor)
        _, pred = torch.max(outputs, 1)

    predicted_idx = pred.item()
    predicted_class = idx_to_class.get(predicted_idx, "Unknown Action")
    print(f"Predicted action for '{video_path}': {predicted_class}")
