import cv2
import mediapipe as mp
import os
from Image_Preprocessing import preprocess_image

# Create a directory to store cropped hand images
output_folder = "Dataset"
os.makedirs(output_folder, exist_ok=True)

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Ask user for the gesture to capture
gesture = input("Enter the gesture to capture: ")

# Create folder for the gesture
gesture_folder = os.path.join(output_folder, gesture)
os.makedirs(gesture_folder, exist_ok=True)

# Initialize image count
image_count = 0

# Capture 50 images for the specified gesture
print(f"Collecting images for {gesture} gesture...")
while image_count < 50:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame
    results = hands.process(frame_rgb)

    # Draw landmarks on hands and save cropped images
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


            # Save the cropped image
            hand_region = frame[y_min:y_max, x_min:x_max]

            # Preprocess the hand region
            final_image = preprocess_image(hand_region)

            # Generate filename
            filename = os.path.join(gesture_folder, f"{gesture}-{image_count}.png")

            cv2.imwrite(filename, final_image)

            # Draw rectangle around the hand
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # Display cropped image
            # cv2.imshow(f'Hand {image_count}', hand_region)

            image_count += 1

    
    # Show the frame
    cv2.imshow('Hand Detection', frame)

    # Check for ESC key press
    key = cv2.waitKey(500)     # delay = 1000 ms
    if key == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Data collection complete!")
