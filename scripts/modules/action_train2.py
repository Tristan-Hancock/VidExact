import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

def main():
    # -------------------- Configuration --------------------
    BASE_DIR = os.path.join("..", "..", "training", "action", "preprocess")
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    VAL_DIR = os.path.join(BASE_DIR, "val")
    BATCH_SIZE = 20
    NUM_FRAMES = 16      # Each .npy file must have shape (16, 112, 112, 3)
    IMG_HEIGHT, IMG_WIDTH = 112, 112
    NUM_CLASSES = 101       # UCF-101

    # -------------------- Load Class Mapping --------------------
    mapping_path = os.path.join(BASE_DIR, "class_to_idx.json")
    with open(mapping_path, "r") as f:
        class_to_idx = json.load(f)
    print("Loaded class mapping (first 5):")
    for k, v in list(class_to_idx.items())[:5]:
        print(f"  {k}: {v}")

    # Create datasets
    train_dataset = UCFActionsDataset(TRAIN_DIR, class_to_idx, max_samples=8000)
    val_dataset = UCFActionsDataset(VAL_DIR, class_to_idx, max_samples=300)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                            num_workers=0)  # Set num_workers to 0 to avoid multiprocessing issues
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                          num_workers=0)  # Set num_workers to 0 to avoid multiprocessing issues

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Improved3DCNN(NUM_CLASSES).to(device)
    print("Model architecture:")
    print(model)
    for name, param in model.named_parameters():
        print(f"Parameter '{name}' is on {param.device}")
        break

    # -------------------- Training Setup --------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    num_epochs = 50

    # -------------------- Training Loop --------------------
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs} starting...", flush=True)
        train_loss, train_acc = train_epoch(train_loader, model, criterion, optimizer, device)
        val_loss, val_acc = evaluate(val_loader, model, criterion, device)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1} complete.", flush=True)
        print(f"  Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}", flush=True)
        print(f"  Val Loss:   {val_loss:.4f}, Val Accuracy:   {val_acc:.2f}", flush=True)

    # -------------------- Save the Model --------------------
    model_save_path = os.path.join("..", "..", "models", "action_recognition_model_final.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}", flush=True)

# -------------------- PyTorch Dataset --------------------
class UCFActionsDataset(Dataset):
    def __init__(self, directory, class_to_idx, max_samples=None):
        self.samples = []
        # Loop through each class folder and collect .npy file paths and labels.
        for cls in os.listdir(directory):
            cls_dir = os.path.join(directory, cls)
            if os.path.isdir(cls_dir):
                for file in os.listdir(cls_dir):
                    if file.endswith(".npy"):
                        file_path = os.path.join(cls_dir, file)
                        label = class_to_idx.get(cls)
                        if label is not None:
                            self.samples.append((file_path, label))
        # Instead of np.random.choice, use random.sample which works on lists.
        if max_samples is not None and len(self.samples) > max_samples:
            self.samples = random.sample(self.samples, max_samples)
        print(f"Found {len(self.samples)} samples in {directory}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        # Load and normalize the video frames
        video = np.load(file_path).astype(np.float32) / 255.0

        # Then apply per-channel mean-std normalization:
        mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        std = np.array([0.225, 0.225, 0.225], dtype=np.float32)

        # video shape is (T, H, W, C). We'll do per-pixel = (value - mean) / std:
        video = (video - mean) / std
        # Data augmentation: random horizontal flip
        if random.random() < 0.5:
            video = video[:, :, ::-1, :]  # Flip width dimension
        video = video.copy()  # Ensure the array has positive strides
        # Convert to torch tensor and rearrange dimensions to (C, T, H, W)
        video = torch.tensor(video).permute(3, 0, 1, 2)
        return video, label



# -------------------- Define the 3D CNN Model --------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- SNIPPET 2: Residual Blocks --------------------
class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Downsample if shape mismatch
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
        out = self.relu(out)
        return out

class Improved3DCNN(nn.Module):
    def __init__(self, num_classes):
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
        
        # Residual blocks for Block 3 & 4
        self.res3 = ResidualBlock3D(128, 256, stride=2)  # from 128->256
        self.res4_1 = ResidualBlock3D(256, 512, stride=2)  # from 256->512
        self.res4_2 = ResidualBlock3D(512, 512, stride=1)  # refine 512 channels

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
        x = self.fc(x)
        return x


def train_epoch(loader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (videos, labels) in enumerate(loader):
        videos, labels = videos.to(device), labels.to(device)
        
        # Check for NaN or Inf in input
        if torch.isnan(videos).any() or torch.isinf(videos).any():
            print(f"Batch {batch_idx} contains NaN or Inf values in videos")
            continue
            
        optimizer.zero_grad()
        outputs = model(videos)
        
        # Check for NaN or Inf in output
        if torch.isnan(outputs).any() or torch.isinf(outputs).any():
            print(f"Batch {batch_idx} contains NaN or Inf values in outputs")
            continue
            
        loss = criterion(outputs, labels)
        
        # Check if loss is NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Batch {batch_idx} contains NaN or Inf loss: {loss.item()}")
            continue
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * videos.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += videos.size(0)
        
        if batch_idx % 10 == 0:
            print(f"  [Batch {batch_idx}] Loss: {loss.item():.4f}", flush=True)
            
    epoch_loss = running_loss / total if total > 0 else 0
    epoch_acc = correct / total if total > 0 else 0
    return epoch_loss, epoch_acc

def evaluate(loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * videos.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += videos.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

if __name__ == '__main__':
    # Call the main function when script is run directly
    main()