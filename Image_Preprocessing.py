import os
import cv2
import numpy as np

# Function to preprocess an image
def preprocess_image(image):
    minValue = 70

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 2)

    # Apply adaptive thresholding to segment the hand region
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, prefinal = cv2.threshold(thresh, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize the hand region to a fixed size if needed
    final = cv2.resize(prefinal, (128, 128))

    # Expand dimensions to make it compatible with model input shape
    return np.expand_dims(final, axis=-1)

'''
# Function to preprocess images in a folder
def preprocess_folder(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each image in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read the image
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            
            # Preprocess the image
            preprocessed_image = preprocess_image(image)
            
            if preprocessed_image is not None:
                # Save the preprocessed image to the output folder
                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, preprocessed_image)
                print(f"Processed: {filename}")
            else:
                print(f"Skipping: {filename}")

# Example usage:
# input_folder = "gesture_images"
input_folder = "Dataset/Xtra"
output_folder = "preprocessed_images"
preprocess_folder(input_folder, output_folder)
'''
