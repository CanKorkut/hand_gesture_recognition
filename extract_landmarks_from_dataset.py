import os
import cv2
import pandas as pd
import mediapipe as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)

landmark_features = []
labels = []

for i in range(5):
    root_path = r"hand_sign_dataset"
    root_path = os.path.join(root_path, str(i+1))
    images = os.listdir(root_path)
    for img_path in tqdm(images):
        image = cv2.imread(os.path.join(root_path, img_path))
        imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                l = []
                for land in landmarks.landmark:
                    l.extend([
                        land.x,
                        land.y,
                        land.z
                    ])
                if len(l) == 42:
                    continue
                else:
                    landmark_features.append(l)
                    labels.append(i+1)


df = pd.DataFrame(landmark_features)
df["label"] = labels
max_count = df['label'].value_counts().max()
balanced_df = df.groupby('label').apply(lambda x: x.sample(max_count, replace=True)).reset_index(drop=True)
shuffled_df = balanced_df.sample(frac=1).reset_index(drop=True)
shuffled_df.to_csv("dataset.csv", index=False)