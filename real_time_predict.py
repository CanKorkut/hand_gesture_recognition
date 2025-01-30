import numpy as np
import torch
import cv2
import mediapipe as mp
import time
import torch.nn as nn


class HandSignCNN(nn.Module):

    def __init__(self, num_classes):

        super(HandSignCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.fc1 = nn.Linear(128 * 59, 128)  # 59, Conv katmanından gelen boyut
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)



    def forward(self, x):

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


model_path = "model.pth"
model = HandSignCNN(5)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

class_item = ["okey", "stop", "up", "forward", "back"]

mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=False,
	                  max_num_hands=2,
	                  min_detection_confidence=0.5,
	                  min_tracking_confidence=0.5)

def predict(rgb_frame):
    model.eval()
    predicted_class = None
    result = hands.process(rgb_frame)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmark_list = []
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])
            input_data = torch.tensor(landmark_list, dtype=torch.float32).view(1, 1, 63)
            with torch.no_grad():
                prediction = model(input_data)
                predicted_class = torch.argmax(prediction).item()
                predicted_class = class_item[predicted_class] 
    return predicted_class


def main():
	cap = cv2.VideoCapture(0)
	mp_draw = mp.solutions.drawing_utils

	pTime = 0
	cTime = 0
	margin_percent = 0.25
	cropped_hand =  np.zeros((256, 256, 3), np.uint8)

	while True:
	    success, img = cap.read()
	    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	    results = hands.process(imgRGB)
	    if results.multi_hand_landmarks:
	            for landmarks in results.multi_hand_landmarks:
	                mp_draw.draw_landmarks(img, landmarks,mp_hand.HAND_CONNECTIONS)
	                # Get the x and y coordinates of the hand landmarks
	                x_min = min([landmark.x for landmark in landmarks.landmark])
	                y_min = min([landmark.y for landmark in landmarks.landmark])
	                x_max = max([landmark.x for landmark in landmarks.landmark])
	                y_max = max([landmark.y for landmark in landmarks.landmark])
	        
	                # Convert normalized coordinates to pixel values
	                h, w, _ = img.shape
	                x_min, y_min = int(x_min * w), int(y_min * h)
	                x_max, y_max = int(x_max * w), int(y_max * h)
	                
	                width = x_max - x_min
	                height = y_max - y_min

	                margin_width = int(width * margin_percent)
	                margin_height = int(height * margin_percent)
	    
	                # Adjust bounding box with margin
	                x_min = max(0, x_min - margin_width)
	                y_min = max(0, y_min - margin_height)
	                x_max = min(w, x_max + margin_width)
	                y_max = min(h, y_max + margin_height)

	        
	                # Crop the image using the bounding box
	                cropped_hand = imgRGB[y_min:y_max, x_min:x_max]
	                result = predict(cropped_hand)
	                
	                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green rectangle
	                cv2.putText(img, str(result), (x_min, y_min - 5),
	                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
	                
	                # Display the cropped hand
	    cv2.imshow("img", img)
	    cv2.imshow("Cropped Hand", cropped_hand)
	    if cv2.waitKey(1) & 0xFF == ord('a'):
	        break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == "__main__":
	main()