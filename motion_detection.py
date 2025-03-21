import cv2
import numpy as np
from deepface import DeepFace

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Read the first frame as the background
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Compute the absolute difference between two frames
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert the difference to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smoothen the image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Thresholding to identify moving objects
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    
    # Dilate the thresholded image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=3)
    
    # Find contours of moving objects
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize variables for bounding box
    x_min, y_min, x_max, y_max = float('inf'), float('inf'), 0, 0
    moving = False  # Flag to check if movement is detected

    for contour in contours:
        if cv2.contourArea(contour) < 500:  # Ignore small movements
            continue

        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)
        moving = True  # Movement detected

    # Ensure valid bounding box values before drawing
    if moving and x_min != float('inf') and y_min != float('inf'):
        # Draw motion bounding box
        cv2.rectangle(frame1, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)

        # Extract face region for emotion detection
        face_roi = frame1[y_min:y_max, x_min:x_max]

        # Perform emotion analysis
        try:
            analysis = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Display the emotion
        text = f"Emotion: {emotion}"
        cv2.putText(frame1, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show the output frame
    cv2.imshow("Motion Detection with Emotion Analysis", frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
