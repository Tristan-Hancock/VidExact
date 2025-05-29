import os
import json
import csv
import wave
import cv2
import numpy as np
import pandas as pd
import ffmpeg
import joblib
import mediapipe as mp
import torch  # For YOLO object detection
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer as KerasInputLayer  # For custom layer
import torch.nn.functional as F
import json
import torch.nn as nn
from PIL import Image
import pytesseract
import json
# Define a custom InputLayer to address the 'batch_shape' issue during model loading
class CustomInputLayer(KerasInputLayer):
    def _init_(self, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super(CustomInputLayer, self)._init_(**kwargs)

# Scene detection imports
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, ThresholdDetector

import vosk


# Global: Load face recognition models (loaded once for efficiency)
vgg_face_descriptor = load_model('../models/vgg_face_descriptor.h5', custom_objects={'InputLayer': CustomInputLayer})
scaler = joblib.load('../models/scaler.pkl')
pca = joblib.load('../models/pca.pkl')
clf = joblib.load('../models/svm_classifier.pkl')
le = joblib.load('../models/label_encoder.pkl')
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if stride!=1 or in_channels!=out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample: identity = self.downsample(x)
        return self.relu(out + identity)


class Improved3DCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d((1,2,2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d((2,2,2))

        self.res3 = ResidualBlock3D(128, 256, stride=2)
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
        x = self.pool4(x).view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)



# Action recognition (PyTorch)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
action_model = Improved3DCNN(num_classes=101).to(device)
action_model.load_state_dict(torch.load('../models/action_recognition_model_9000_50.pth', map_location=device))
action_model.eval()

with open('../models/class_to_idx.json','r') as f:
    class_to_idx = json.load(f)
idx_to_class = {v:k for k,v in class_to_idx.items()}


# Function 4: Speech-to-Text using Vosk (with ffmpeg for audio extraction)
def speech_to_text(clip_path, model_path="model"):
    temp_audio = "temp_audio.wav"
    
    try:
        ffmpeg.input(clip_path).output(temp_audio, ac=1, ar='16k').run(overwrite_output=True)
    except ffmpeg.Error as e:
        print("Error extracting audio:", e)
        return ""
    
    wf = wave.open(temp_audio, "rb")
    print("[LOG] speech_to_text: Checking audio format...")
    print("    Channels:", wf.getnchannels())
    print("    Sample Width:", wf.getsampwidth())
    print("    Frame Rate:", wf.getframerate())
    
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != 16000:
        wf.close()
        os.remove(temp_audio)
        raise ValueError("Audio must be WAV mono PCM at 16kHz.")
    
    model_instance = vosk.Model(lang="en-us") if model_path == "model" else vosk.Model(model_path)
    rec = vosk.KaldiRecognizer(model_instance, wf.getframerate())
    
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result_json = json.loads(rec.Result())
            results.append(result_json)
    
    final_result = json.loads(rec.FinalResult())
    results.append(final_result)
    wf.close()
    os.remove(temp_audio)
    
    recognized_texts = [r["text"] for r in results if "text" in r]
    full_text = " ".join(recognized_texts)
    print(f"[LOG] speech_to_text: Transcribed {len(full_text.split())} word(s).")
    return full_text


# Function 5: Face Detection & Recognition using MediaPipe
def recognize_face_mediapipe(frame_path):
    # Initialize MediaPipe face detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    
    image = cv2.imread(frame_path)
    if image is None:
        print("[LOG] recognize_face_mediapipe: Could not read image from", frame_path)
        return "NoFaceDetected"
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image_rgb)
    
    if not results.detections:
        print("[LOG] recognize_face_mediapipe: No face detected in the frame.")
        return "NoFaceDetected"
    
    # Use the first detected face
    detection = results.detections[0]
    bboxC = detection.location_data.relative_bounding_box
    ih, iw, _ = image.shape
    x_min = max(0, int(bboxC.xmin * iw))
    y_min = max(0, int(bboxC.ymin * ih))
    width = int(bboxC.width * iw)
    height = int(bboxC.height * ih)
    face_snippet = image[y_min:y_min+height, x_min:x_min+width]
    
    identity = recognize_face_from_snippet(face_snippet)
    return identity

def recognize_face_from_snippet(face_snippet):
    global vgg_face_descriptor, scaler, pca, clf, le
    # Preprocess the face snippet and perform recognition
    img = cv2.cvtColor(face_snippet, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img_expanded = np.expand_dims(img, axis=0)
    
    embedding_vector = vgg_face_descriptor.predict(img_expanded)[0]
    embedding_vector = scaler.transform([embedding_vector])
    embedding_vector = pca.transform(embedding_vector)
    y_pred = clf.predict(embedding_vector)
    predicted_name = le.inverse_transform(y_pred)[0]
    return predicted_name


# Function 6: Object Detection using YOLOv5
def detect_objects_yolo(image_path, conf_threshold=0.5):
    image = cv2.imread(image_path)
    if image is None:
        print(f"[LOG] detect_objects_yolo: Could not read image from {image_path}")
        return []
    
    results = yolo_model(image)
    df = results.pandas().xyxy[0]
    df = df[df['confidence'] >= conf_threshold]
    
    img_area = image.shape[0] * image.shape[1]
    detections = []
    for index, row in df.iterrows():
        x1, y1, x2, y2 = row['xmin'], row['ymin'], row['xmax'], row['ymax']
        bbox_area = (x2 - x1) * (y2 - y1)
        prominence = bbox_area / img_area
        detections.append({
            "class": row['name'],
            "confidence": float(row['confidence']),
            "prominence": prominence
        })
    
    print(f"[LOG] detect_objects_yolo: Found {len(detections)} high-confidence objects.")
    return detections
# Function 7: Action detection
def recognize_action(clip_path, model, idx_to_class, device, object_detections, threshold=0.6, margin=0.15):
    cap = cv2.VideoCapture(clip_path)
    frames, total = [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, total-1, 16, dtype=int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        frames.append(cv2.resize(frame, (112,112)))
    cap.release()

    if len(frames) < 16:
        frames += [frames[-1]]*(16-len(frames))

    video = (np.stack(frames).astype(np.float32)/255.0 - 0.45) / 0.225
    tensor = torch.tensor(video, dtype=torch.float32).permute(3,0,1,2).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1).cpu().numpy()[0]
    top_idx, top_conf = probs.argmax(), probs.max()
    runner_up = sorted(probs)[-2]

    label = idx_to_class[top_idx]
    if top_conf < threshold or (top_conf - runner_up) < margin:
        label = "Unknown"

    # Contextual sports filter
    classes = {d["class"] for d in object_detections}
    sports_map = {
        "TennisSwing": {"sports ball", "tennis racket"},
        "BaseballPitch": {"baseball bat"},
        "GolfSwing": {"golf club"},
    }
    if label in sports_map and not (sports_map[label] & classes):
        label = "Unknown"

    return label, float(top_conf)

#Function 8 - Character Recognition
def recognize_text(image_path):
    try:
        # Open the image using PIL and run pytesseract OCR
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        print(f"[LOG] recognize_text: OCR found {len(text.split())} words.")
        return text.strip()
    except Exception as e:
        print("[LOG] recognize_text: Error processing OCR:", e)
        return ""
    


import os
import json
import cv2

import os
import json
import csv
import cv2

def process_video(video_path):
    # ———————————————————————————————————————————————————————
    # 1) Resolve directories
    # ———————————————————————————————————————————————————————
    script_dir  = os.path.dirname(os.path.abspath(__file__))  # .../backend/app
    project_root= os.path.dirname(script_dir)                 # .../backend
    output_dir  = os.path.join(project_root, "output")
    frames_dir  = os.path.join(output_dir, "frames")
    csv_dir     = os.path.join(output_dir, "csv")
    json_dir    = os.path.join(output_dir, "json")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)

    csv_path  = os.path.join(csv_dir,  "final_results.csv")
    json_path = os.path.join(json_dir, "final_results.json")

    print(f"[LOG] process_video: Opening '{video_path}'")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = int(total_frames / fps)

    # ———————————————————————————————————————————————————————
    # 2) Speech-to-text for entire video
    # ———————————————————————————————————————————————————————
    caption = speech_to_text(video_path)

    results = []
    # ———————————————————————————————————————————————————————
    # 3) Sample one frame per second
    # ———————————————————————————————————————————————————————
    for sec in range(duration_s):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            print(f"[WARN] could not read frame at {sec}s")
            continue

        frame_name     = f"Frame{sec+1}"
        frame_file     = frame_name + ".jpg"
        frame_path     = os.path.join(frames_dir, frame_file)
        cv2.imwrite(frame_path, frame)

        # ———————————————————————————————————————————————————————
        # 4) Run analyses
        # ———————————————————————————————————————————————————————
        try:
            face_id = recognize_face_mediapipe(frame_path)
        except Exception as e:
            print(f"[ERROR] Face detection failed at {frame_name}: {e}")
            face_id = "Error"

        try:
            objects = detect_objects_yolo(frame_path)
        except Exception as e:
            print(f"[ERROR] Object detection failed at {frame_name}: {e}")
            objects = []

        try:
            ocr_txt = recognize_text(frame_path)
        except Exception as e:
            print(f"[ERROR] OCR failed at {frame_name}: {e}")
            ocr_txt = ""

        try:
            action_lbl, action_conf = recognize_action(
                video_path, action_model, idx_to_class, device, objects
            )
        except Exception as e:
            print(f"[ERROR] Action recognition failed at {frame_name}: {e}")
            action_lbl, action_conf = "Error", 0.0

        timestamp = sec
        print(f"[LOG] {frame_name} @ {timestamp}s | face={face_id} "
              f"| objs={len(objects)} | words={len(ocr_txt.split())}")

        results.append({
            "frame_name":        frame_name,
            "face_detected":     face_id,
            "caption":           caption,
            "object_detection":  objects,
            "action":            action_lbl,
            "action_confidence": action_conf,
            "ocr_text":          ocr_txt,
            "timestamp_start":   timestamp,
            "timestamp_end":     timestamp
        })

    cap.release()

    # ———————————————————————————————————————————————————————
    # 5) Write CSV
    # ———————————————————————————————————————————————————————
    fieldnames = [
        "frame_name","face_detected","caption",
        "object_detection","action","action_confidence",
        "ocr_text","timestamp_start","timestamp_end"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"[LOG] process_video: Saved CSV to '{csv_path}'")

    # ———————————————————————————————————————————————————————
    # 6) Write JSON
    # ———————————————————————————————————————————————————————
    os.makedirs(json_dir, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump({"results": results}, jf, indent=2)
    print(f"[LOG] process_video: Also saved JSON to '{json_path}'")






# if __name__ == "__main__":
#     print("[DEBUG] Starting main pipeline process...")
#     input_video = "../input/Video4.mp4"
#     process_video(input_video)
from pathlib import Path

if __name__ == "__main__":
    # 1) Locate this script
    script_path = Path(__file__).resolve()
    print(f"[DEBUG] Script path: {script_path}")

    # 2) Build the input-video path
    project_root = script_path.parent.parent  # .../backend
    input_dir    = project_root / "input"
    input_video  = (input_dir / "Video4.mp4").resolve()

    print(f"[DEBUG] Looking for video at: {input_video}")
    print(f"[DEBUG] Input folder contents: {[p.name for p in input_dir.iterdir()]}")

    # 3) Check & fail fast
    if not input_video.is_file():
        raise FileNotFoundError(
            f"[ERROR] No 'Video4.mp4' in {input_dir}\n"
            f"Directory contains: {[p.name for p in input_dir.iterdir()]}"
        )

    # 4) All good—run
    print("[DEBUG] Video found, starting pipeline…")
    process_video(str(input_video))
    print("[DEBUG] Done.")

