import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize hand detector
detector = HandDetector(detectionCon=0.8)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Read frame from webcam
    success, img = cap.read()
    
    if not success:
        print("Failed to read from webcam")
        break
    
    # Convert the image to RGB (cvzone requires RGB format)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Find hands
    imgRGB, hands = detector.findHands(imgRGB)
    
    if hands and len(hands) > 0:  # Check if hands is not empty and contains at least one hand
        # Loop through each hand
        for hand in hands:
            # Get the bounding box of the hand
            bbox = hand["bbox"]
            # Extract bounding box coordinates
            x, y, w, h = bbox
            # Draw rectangle around the hand
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    # Display the image
    cv2.imshow("Hand Tracking", img)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
