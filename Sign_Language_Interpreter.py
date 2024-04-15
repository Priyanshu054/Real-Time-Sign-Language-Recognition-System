import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
# from Image_Preprocessing import preprocess_image

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the trained model
model = load_model('Trained_Model/CNN_Model.keras')

gesture = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'Good', 'H', 'Hello', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

def preprocess_image(image):
    minValue = 0    # 70
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)    # (gray, (5,5), 2)

    # Apply adaptive thresholding to segment the hand region
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, prefinal = cv2.threshold(thresh, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    final = cv2.resize(prefinal, (128, 128))
    return np.expand_dims(final, axis=-1)

def detect_hand(frame):
    # Process the frame
    results = hands.process(frame)
    
    if results.multi_hand_landmarks:
        return True, results.multi_hand_landmarks[0].landmark
    else:
        return False, None

def extract_hand_region(frame, landmarks, padding=20):
    x_values = [landmark.x for landmark in landmarks]
    y_values = [landmark.y for landmark in landmarks]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_min = max(0, int(x_min * frame.shape[1]) - padding)
    x_max = min(frame.shape[1], int(x_max * frame.shape[1]) + padding)
    y_min = max(0, int(y_min * frame.shape[0]) - padding)
    y_max = min(frame.shape[0], int(y_max * frame.shape[0]) + padding)

    return frame[y_min:y_max, x_min:x_max], (x_min, y_min, x_max, y_max)

def recognize_gesture(hand_region):
    final_image = preprocess_image(hand_region)
    predictions = model.predict(final_image[np.newaxis, ...])
    predicted_label = np.argmax(predictions)
    confidence = predictions[0][predicted_label]

    if confidence > 0.4:
        return gesture[predicted_label], confidence
    else:
        return "Unknown", confidence

def draw_on_frame(frame, hand_region_coords, label, confidence):
    x_min, y_min, x_max, y_max = hand_region_coords
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"{confidence:.2f}", (x_min, y_max+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hand_detected, landmarks = detect_hand(frame)
    if hand_detected:
        hand_region, hand_region_coords = extract_hand_region(frame, landmarks, padding=20)

        x_min, y_min, x_max, y_max = hand_region_coords
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        gesture_label, confidence = recognize_gesture(hand_region)
        draw_on_frame(frame, hand_region_coords, gesture_label, confidence)
    else:
        cv2.putText(frame, "No Hand Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Sign Language Interpreter', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:    # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
