import cv2
import numpy as np
from tensorflow.keras.models import load_model

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

def recognize_gesture(hand_region):
    final_image = preprocess_image(hand_region)
    cv2.imshow('hand', final_image)     ##################
    predictions = model.predict(final_image[np.newaxis, ...])
    predicted_label = np.argmax(predictions)
    confidence = predictions[0][predicted_label]

    if confidence > 0.6:
        return gesture[predicted_label], confidence
    else:
        return "unknown", confidence

def draw_on_frame(frame, hand_region_coords, label, confidence):
    x_min, y_min, x_max, y_max = hand_region_coords
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    cv2.putText(frame, label, (x_min, y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"{confidence:.2f}", (x_min, y_max+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Define the region for gesture recognition
    
    # gesture_region_coords = (340, 50, 590, 300) # (250,250)
    gesture_region_coords = (350, 50, 590, 270) # (220,220)
    x_min, y_min, x_max, y_max = gesture_region_coords
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

    # Crop the region for gesture recognition
    gesture_region = frame[y_min:y_max, x_min:x_max]

    # Recognize gesture in the cropped region
    gesture_label, confidence = recognize_gesture(gesture_region)

    # Draw the result on the frame
    draw_on_frame(frame, gesture_region_coords, gesture_label, confidence)

    cv2.imshow('Sign Language Interpreter', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:  # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
