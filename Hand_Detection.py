#%%
import cv2
import mediapipe as mp
from Image_Preprocessing import preprocess_image

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(frame_rgb)

    # Draw landmarks on hands and preprocess the hand region
    if results.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Calculate bounding box coordinates with padding
            landmarks = hand_landmarks.landmark
            x_values = [landmark.x for landmark in landmarks]
            y_values = [landmark.y for landmark in landmarks]
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
            padding = 20  # Adjust this value as needed
            x_min = max(0, int(x_min * frame.shape[1]) - padding)
            x_max = min(frame.shape[1], int(x_max * frame.shape[1]) + padding)
            y_min = max(0, int(y_min * frame.shape[0]) - padding)
            y_max = min(frame.shape[0], int(y_max * frame.shape[0]) + padding)

            # Extract the hand region
            hand_region = frame[y_min:y_max, x_min:x_max]

            # Preprocess the hand region
            final_image = preprocess_image(hand_region)

            # Draw rectangle around the hand in the original frame
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Show the preprocessed hand region
            cv2.imshow('Preprocessed Hand Region', final_image)

    # Show the original frame
    cv2.imshow('Real-time Hand Detection', frame)

    # Check for ESC key press
    key = cv2.waitKey(1)
    if key == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# %%
