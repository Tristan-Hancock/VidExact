import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
current_dir = os.path.dirname(os.path.abspath(__file__))
class_to_idx_path = os.path.join(current_dir, "..","..", "..", "training", "action", "preprocess", "class_to_idx.json")

# -------------------- Model Definition --------------------
# -------------------- New Residual Block Model Definition --------------------
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

class Improved3DCNN(nn.Module):
    def __init__(self, num_classes=101):
        super(Improved3DCNN, self).__init__()
        # Block 1: Initial convolution
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        # Block 2: Increase depth
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Block 3: Residual block from 128 -> 256 (with downsampling)
        self.res3 = ResidualBlock3D(128, 256, stride=2)
        
        # Block 4: Two residual blocks to reach 512 channels
        self.res4_1 = ResidualBlock3D(256, 512, stride=2)
        self.res4_2 = ResidualBlock3D(512, 512, stride=1)
        
        self.pool4 = nn.AdaptiveAvgPool3d(1)  # Global pooling
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Input x: (B, 3, T, H, W)
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.res3(x)
        x = self.res4_1(x)
        x = self.res4_2(x)
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)



# -------------------- Dataset Class for Test Data --------------------
class UCFActionsTestDataset(Dataset):
    def __init__(self, directory, class_to_idx):
        self.samples = []
        for cls in os.listdir(directory):
            cls_dir = os.path.join(directory, cls)
            if os.path.isdir(cls_dir):
                for file in os.listdir(cls_dir):
                    if file.endswith(".npy"):
                        file_path = os.path.join(cls_dir, file)
                        label = class_to_idx.get(cls)
                        if label is not None:
                            self.samples.append((file_path, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        video = np.load(file_path).astype(np.float32) / 255.0
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
        video = (video - mean) / std
        # Convert to tensor and re-arrange (C, T, H, W)
        video = torch.tensor(video).permute(3, 0, 1, 2)
        return video, label


# -------------------- Load Model and Mappings --------------------
num_classes = 101  # UCF-101
model = Improved3DCNN(num_classes=num_classes).to(device)
checkpoint_path = os.path.join("..","..","models", "action_recognition_model_9000_50.pth")
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
print("Model loaded from", checkpoint_path)

# Load class mapping
with open(class_to_idx_path, "r") as f:
    class_to_idx = json.load(f)
# Invert mapping to get index->class name
idx_to_class = {v: k for k, v in class_to_idx.items()}
print("Class mapping loaded.")


# -------------------- Evaluate on Test Set --------------------
def evaluate(loader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for videos, labels in loader:
            videos = videos.to(device)
            labels = labels.to(device)
            outputs = model(videos)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)

test_dir = os.path.join("..","..", "..", "training", "action", "preprocess", "test")
test_dataset = UCFActionsTestDataset(test_dir, class_to_idx)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Evaluating on test set...")
preds, labels = evaluate(test_loader, model, device)
accuracy = (preds == labels).mean()
print(f"Test Accuracy: {accuracy:.2f}")

cm = confusion_matrix(labels, preds)
print("Confusion Matrix:")
print(cm)

print("Classification Report:")
report = classification_report(labels, preds, target_names=[idx_to_class[i] for i in range(num_classes)])
print(report)

# Optional: Visualize confusion matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()


# -------------------- Inference on a Real-World Video --------------------
def extract_frames_from_video(video_path, num_frames=16, img_size=(112, 112)):
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
            frames.append(frame)
        frame_idx += 1
    cap.release()
    if len(frames) < num_frames and len(frames) > 0:
        last_frame = frames[-1]
        while len(frames) < num_frames:
            frames.append(last_frame)
    if len(frames) < num_frames:
        return None
    # Convert (T, H, W, C) to (C, T, H, W)
    video_np = np.array(frames)
    video_np = np.transpose(video_np, (3, 0, 1, 2))
    return torch.tensor(video_np, dtype=torch.float32)

video_snippet_path = "../../input/Video4.mp4"  # Path to your 4-second video snippet
video_tensor = extract_frames_from_video(video_snippet_path, num_frames=16, img_size=(112, 112))
if video_tensor is None:
    print("Failed to process video snippet.")
else:
    video_tensor = video_tensor.unsqueeze(0).to(device)  # (1, 3, T, H, W)
    with torch.no_grad():
        outputs = model(video_tensor)
        _, pred = torch.max(outputs, 1)
    predicted_class = idx_to_class[pred.item()]
    print(f"Predicted action for video '{video_snippet_path}': {predicted_class}")
