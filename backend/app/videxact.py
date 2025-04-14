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
vgg_face_descriptor = load_model('../models/vgg_face_descriptor.h5')
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

with open('../../training/action/preprocess/class_to_idx.json','r') as f:
    class_to_idx = json.load(f)
idx_to_class = {v:k for k,v in class_to_idx.items()}

# Function 1: Detect Scenes using scenedetect
def detect_scenes(video_path, content_threshold=50.0, threshold_val=20, min_scene_len=15):
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=content_threshold))
    scene_manager.add_detector(ThresholdDetector(threshold=threshold_val, min_scene_len=min_scene_len))
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()
    print(f"[LOG] detect_scenes: Detected {len(scene_list)} scene(s) in '{video_path}'.")
    return scene_list


# Function 2: Trim Video Clip using ffmpeg-python
def trim_clip(input_video, start_time, end_time, clip_index, max_duration=4.0, output_dir='clips'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    duration = end_time - start_time
    if duration > max_duration:
        end_time = start_time + max_duration  # Truncate to max_duration
    clip_name = f"Scene_{clip_index:03d}.mp4"
    output_path = os.path.join(output_dir, clip_name)
    
    try:
        ffmpeg.input(input_video, ss=start_time, t=(end_time - start_time)) \
              .output(output_path, vcodec='libx264', acodec='aac') \
              .run(overwrite_output=True)
    except ffmpeg.Error as e:
        print("Error trimming clip:", e)
    
    print(f"[LOG] trim_clip: Trimmed clip {clip_name} from {start_time}s to {end_time}s (max {max_duration}s).")
    return output_path


# Function 3: Extract Key Frame using OpenCV
def extract_key_frame(clip_path, frame_index='middle', output_dir='frames'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    
    # Determine target time: use the middle of the clip by default
    target_time = duration / 2 if frame_index == 'middle' else 0
    target_frame = int(target_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    if ret:
        frame_file_name = os.path.basename(clip_path).split('.')[0] + "_keyframe.jpg"
        frame_path = os.path.join(output_dir, frame_file_name)
        cv2.imwrite(frame_path, frame)
        print(f"[LOG] extract_key_frame: Extracted key frame at {target_time:.2f}s -> {frame_path}")
        cap.release()
        return frame_path
    else:
        cap.release()
        print("[LOG] extract_key_frame: Failed to extract frame.")
        return None


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
    


def process_video(video_path):
    print(f"[LOG] process_video: Starting processing for '{video_path}'...")
    
    # 1) Scene Detection
    scenes = detect_scenes(video_path, content_threshold=10.0, threshold_val=20, min_scene_len=15)
    print(f"[DEBUG] process_video: Scenes -> {scenes}")
    if not scenes:
        print("[DEBUG] process_video: No scenes detected! The CSV will likely be empty.")
    
    results = []
    
    for idx, scene in enumerate(scenes, start=1):
        start_sec = scene[0].get_seconds()
        end_sec = scene[1].get_seconds()
        print(f"[LOG] process_video: Processing scene #{idx} | Start: {start_sec:.2f}s, End: {end_sec:.2f}s.")
        
        # 2) Trim the scene using ffmpeg
        clip_path = trim_clip(video_path, start_sec, end_sec, idx, max_duration=4.0, output_dir='clips')
        
        # 3) Extract a key frame from the trimmed clip using OpenCV
        frame_path = extract_key_frame(clip_path, frame_index='middle', output_dir='frames')
        ocr_text = recognize_text(frame_path) if frame_path else ""
        
        # 4) Speech-to-Text from the trimmed clip
        recognized_speech = speech_to_text(clip_path)
        
        # 5) Face detection & recognition on the key frame using MediaPipe logic
        face_identity = recognize_face_mediapipe(frame_path)
        
        # 6) Object detection on the key frame using YOLOv5
        object_detections = detect_objects_yolo(frame_path)
      

        # 7) Collect and store the results (object detections stored as a JSON string)
        record = {
            "clip_name": os.path.basename(clip_path),
            "frame_name": os.path.basename(frame_path) if frame_path else "None",
            "face_detected": face_identity,
            "caption": recognized_speech,
            "object_detection": json.dumps(object_detections),
            "ocr_text": ocr_text,
            "timestamp_start": start_sec,
            "timestamp_end": end_sec
        }
        # Add action prediction after record exists
        action_label, action_conf = recognize_action(clip_path, action_model, idx_to_class, device, object_detections)
        record["action"] = action_label
        record["action_confidence"] = action_conf
        results.append(record)
        print(f"[LOG] process_video: Scene #{idx} processing complete.\n")
    
    print(f"[DEBUG] process_video: Final results -> {results}")
    
    # Write results to CSV
    csv_dir= os.path.join("..","output","csv")
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    csv_filename = os.path.join(csv_dir, "final_results.csv")
    fieldnames = ["clip_name","frame_name","face_detected","caption","object_detection","action","action_confidence","ocr_text","timestamp_start","timestamp_end"]
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"[LOG] process_video: All scenes processed. Results saved to '{csv_filename}'.")


if __name__ == "__main__":
    print("[DEBUG] Starting main pipeline process...")
    input_video = "../input/Video4.mp4"
    process_video(input_video)
