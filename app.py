import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
from cnn import HandSignCNN
import torch

# Flask uygulaması başlatma
app = Flask(__name__)

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

model_path = "model.pth"
model = HandSignCNN(5)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

class_item = ["okey", "stop", "up", "forward", "back"]

# El landmarklarını işleme fonksiyonu
def get_landmarks(image):
    # Resmi RGB formatına dönüştür
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # MediaPipe ile elde edilen el landmarks
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        return landmarks
    else:
        return None

# Tahmin fonksiyonu
def predict_gesture(landmarks):
    model.eval()
    predicted_class = None
    input_data = torch.tensor(landmarks, dtype=torch.float32).view(1, 1, 63)
    with torch.no_grad():
        prediction = model(input_data)
        predicted_class = torch.argmax(prediction).item()
        predicted_class = class_item[predicted_class] 
    return predicted_class


# Flask route, resim upload ve tahmin
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream)
    
    # OpenCV'ye uygun formatta resmi yükleyin
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Landmarkları al
    landmarks = get_landmarks(img)
    
    if landmarks is None:
        return jsonify({'error': 'Hand not detected'}), 400

    # Gesture tahmini yap
    gesture = predict_gesture(landmarks)
    
    return jsonify({'gesture': gesture})

if __name__ == '__main__':
    app.run(debug=True)
