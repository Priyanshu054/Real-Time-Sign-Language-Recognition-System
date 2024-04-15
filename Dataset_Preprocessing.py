import os
import cv2
import numpy as np

# Function to preprocess an image
def preprocess_image(image):
    minValue = 0    # 70

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blur = cv2.GaussianBlur(gray, (5, 5), 0)    # (gray, (5,5), 2)

    # Apply adaptive thresholding to segment the hand region
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    _, prefinal = cv2.threshold(thresh, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Resize the hand region to a fixed size if needed
    final = cv2.resize(prefinal, (128, 128))

    # Expand dimensions to make it compatible with model input shape
    return np.expand_dims(final, axis=-1)

# Function to preprocess images in a folder and save them in a new folder
def preprocess_folder(input_folder, output_folder, limit_per_folder=800):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Loop through each folder in the input folder
    for folder_name in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder_name)
        if os.path.isdir(folder_path):
            output_folder_path = os.path.join(output_folder, folder_name)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            # Count the number of processed images in the folder
            processed_count = 0
            # Loop through each image in the folder
            for i, filename in enumerate(os.listdir(folder_path)):
                if processed_count >= limit_per_folder:
                    break
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    # Read the image
                    image_path = os.path.join(folder_path, filename)
                    image = cv2.imread(image_path)
                    # Preprocess the image
                    preprocessed_image = preprocess_image(image)
                    # Save the preprocessed image to the output folder with the folder name prefix
                    output_filename = f"{folder_name}-{i+1:03d}.jpg"
                    output_path = os.path.join(output_folder_path, output_filename)
                    cv2.imwrite(output_path, preprocessed_image)
                    print(f"Processed: {filename} -> {output_filename}")
                    processed_count += 1

# Example usage:
input_folder = "Sign_Language_Dataset"
output_folder = "New_Dataset"
preprocess_folder(input_folder, output_folder, limit_per_folder=500)
