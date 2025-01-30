import cv2
import mediapipe as mp
import os
import uuid

for i in range(5):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    video_path = str(i+1) + '.avi' 
    cap = cv2.VideoCapture(video_path)

    output_folder = str(i+1)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_count = 0

    margin_percent = 0.25

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Get the x and y coordinates of the hand landmarks
                x_min = min([landmark.x for landmark in landmarks.landmark])
                y_min = min([landmark.y for landmark in landmarks.landmark])
                x_max = max([landmark.x for landmark in landmarks.landmark])
                y_max = max([landmark.y for landmark in landmarks.landmark])

                # Convert normalized coordinates to pixel values
                h, w, _ = frame.shape
                x_min, y_min = int(x_min * w), int(y_min * h)
                x_max, y_max = int(x_max * w), int(y_max * h)

                # Add margin (15%) around the bounding box
                width = x_max - x_min
                height = y_max - y_min

                margin_width = int(width * margin_percent)
                margin_height = int(height * margin_percent)

                # Adjust bounding box with margin
                x_min = max(0, x_min - margin_width)
                y_min = max(0, y_min - margin_height)
                x_max = min(w, x_max + margin_width)
                y_max = min(h, y_max + margin_height)

                # Crop the hand from the frame with the margin
                cropped_hand = frame[y_min:y_max, x_min:x_max]

                # Generate a unique filename for each cropped image
                unique_filename = str(uuid.uuid4()) + '.jpg'

                # Save the cropped hand to the output folder
                cv2.imwrite(os.path.join(output_folder, unique_filename), cropped_hand)
                print(f"Saved cropped hand as {unique_filename}")

    cap.release()
    cv2.destroyAllWindows()

    hands.close()
