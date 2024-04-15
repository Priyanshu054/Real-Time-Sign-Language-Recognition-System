#%%
import cv2
import os
from Image_Preprocessing import preprocess_image

# Create a directory to store cropped hand images
output_folder = "Dataset"
os.makedirs(output_folder, exist_ok=True)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Define region of interest (ROI) for hand detection
roi_top_left = (350, 50)  # Define the top-left corner of ROI
roi_bottom_right = (590, 270)  # Define the bottom-right corner of ROI

# Ask user for the gesture to capture
gesture = input("Enter the gesture to capture: ")

# Create folder for the gesture
gesture_folder = os.path.join(output_folder, gesture)
os.makedirs(gesture_folder, exist_ok=True)

# Initialize image count
image_count = 0

# Capture images for the specified gesture
print(f"Collecting images for {gesture} gesture...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Draw ROI rectangle
    cv2.rectangle(frame, roi_top_left, roi_bottom_right, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Hand Detection', frame)

    key = cv2.waitKey(30)

    # Press 's' to capture hand image
    if key == ord('s'):
        # Crop hand region
        hand_region = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]

        # Preprocess the hand region
        final_image = preprocess_image(hand_region)

        # Generate filename
        filename = os.path.join(gesture_folder, f"{gesture}-{image_count}.png")

        # Save the cropped image
        cv2.imwrite(filename, final_image)

        print(f"Image {image_count} saved.")
        image_count += 1

    # Press 'Esc' to exit
    elif key == 27:  # ESC key
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("Data collection complete!")
# %%