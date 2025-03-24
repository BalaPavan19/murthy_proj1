from flask import Flask, request, render_template
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from mtcnn import MTCNN
import os

app = Flask(__name__)

# Load pre-trained ResNet-50 as feature extractor
resnet = models.resnet50(pretrained=True)
resnet.eval()
resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove final layer

# Simple classifier for demo
class DeepfakeClassifier(nn.Module):
    def __init__(self, input_size=2048):
        super(DeepfakeClassifier, self).__init__()
        self.fc = nn.Linear(input_size, 2)  # Real or Fake
    
    def forward(self, x):
        return self.fc(x)

classifier = DeepfakeClassifier()
classifier.eval()

# Face detector
detector = MTCNN()

# Extract faces from video
def extract_faces_from_video(video_path, max_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(frame_rgb)
        if faces:
            x, y, w, h = faces[0]['box']
            face = frame_rgb[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            frames.append(face)
        
        frame_count += 1
    
    cap.release()
    return frames

# Preprocess frame for ResNet
def preprocess_frame(frame):
    frame = torch.tensor(frame, dtype=torch.float32).permute(2, 0, 1)  # HWC to CHW
    frame = frame / 255.0
    frame = frame.unsqueeze(0)
    return frame

# Predict if video is deepfake
def predict_deepfake(video_path):
    faces = extract_faces_from_video(video_path)
    if not faces:
        return "No faces detected.", None
    
    features = []
    for face in faces:
        input_tensor = preprocess_frame(face)
        with torch.no_grad():
            feature = resnet(input_tensor).flatten()
            features.append(feature)
    
    avg_features = torch.mean(torch.stack(features), dim=0)
    with torch.no_grad():
        output = classifier(avg_features)
        probabilities = torch.softmax(output, dim=0)
        prediction = torch.argmax(probabilities).item()
    
    result = "Fake" if prediction == 1 else "Real"
    return result, probabilities

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', message="No file selected.")
        
        if file:
            video_path = os.path.join("uploads", file.filename)
            if not os.path.exists("uploads"):
                os.makedirs("uploads")
            file.save(video_path)
            
            result, probs = predict_deepfake(video_path)
            if probs is not None:
                message = f"Prediction: {result} (Real: {probs[0]:.4f}, Fake: {probs[1]:.4f})"
            else:
                message = result
            
            os.remove(video_path)  # Clean up
            return render_template('index.html', message=message)
    
    return render_template('index.html', message=None)

if __name__ == '__main__':
    app.run(debug=True)