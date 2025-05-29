import os
import cv2
import numpy as np
import random
import json

# -------------------- Configuration --------------------
DATASET_DIR = "../datasets/UCF-101"              # Updated dataset path
OUTPUT_DIR = "../training/action/preprocess"     # Preprocessed .npy save path
FRAME_COUNT = 16                                     # Frames per video
IMG_SIZE = (112, 112)                                # Resized frame dimensions

# Train/Val/Test split ratios
SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}

# -------------------- Utility Functions --------------------
def create_output_dirs(class_names):
    """Create train/val/test dirs for each class."""
    for split in SPLIT_RATIOS:
        for cls in class_names:
            out_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(out_dir, exist_ok=True)

def extract_frames(video_path, num_frames=FRAME_COUNT, img_size=IMG_SIZE):
    """Extract and preprocess uniformly sampled frames from a video."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Warning: No frames in {video_path}")
        cap.release()
        return np.array([])

    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames, frame_idx, ret = [], 0, True

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

    if len(frames) < num_frames and frames:
        frames += [frames[-1]] * (num_frames - len(frames))
    
    return np.array(frames, dtype=np.float32)

# -------------------- Main Preprocessing Script --------------------
def main():
    classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])
    print(f"Found {len(classes)} classes: {classes}")

    create_output_dirs(classes)

    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "class_to_idx.json"), "w") as f:
        json.dump(class_to_idx, f, indent=4)

    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        videos = [f for f in os.listdir(cls_dir) if f.endswith(".avi")]
        random.shuffle(videos)
        num_videos = len(videos)

        train_cut = int(SPLIT_RATIOS["train"] * num_videos)
        val_cut = train_cut + int(SPLIT_RATIOS["val"] * num_videos)
        splits = {
            "train": videos[:train_cut],
            "val": videos[train_cut:val_cut],
            "test": videos[val_cut:]
        }

        for split, vid_list in splits.items():
            for vid in vid_list:
                video_path = os.path.join(cls_dir, vid)
                frames = extract_frames(video_path)
                if frames.size == 0:
                    continue
                out_filename = os.path.splitext(vid)[0] + ".npy"
                out_path = os.path.join(OUTPUT_DIR, split, cls, out_filename)
                np.save(out_path, frames)
                print(f"Saved {out_path} with shape {frames.shape}")

if __name__ == "__main__":
    main()
