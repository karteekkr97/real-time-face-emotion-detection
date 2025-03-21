import cv2
import numpy as np
from deepface import DeepFace

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam in fullscreen mode
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# Emotion categories
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Store gender and stable age range
stored_gender = None
stable_age = None  # Store first detected age to stabilize fluctuations

# Function to draw vertical emotion bar chart inside webcam feed
def draw_vertical_emotion_chart(frame, emotion_scores, x, y, width=250, height=400):
    bar_height = height // len(emotions)  # Height for each bar
    max_width = width - 70  # Maximum width for strongest emotion
    padding = 15  # Space for labels

    # Ensure the background fully covers text and bars
    chart_width = max_width + 150
    chart_height = height + 50

    # Draw black background for chart
    cv2.rectangle(frame, (x, y), (x + chart_width, y + chart_height), (0, 0, 0), -1)

    for i, emotion in enumerate(emotions):
        value = int(emotion_scores.get(emotion.lower(), 0) * max_width / 100)  # Normalize values
        bar_x = x + 20
        bar_y = y + i * bar_height + 30

        # Draw white emotion bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + value, bar_y + bar_height - 5), (255, 255, 255), -1)

        # Draw emotion label next to the bar
        cv2.putText(frame, emotion, (bar_x + max_width + 40, bar_y + bar_height - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        # Draw rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the detected face
        face_crop = frame[y:y + h, x:x + w]

        try:
            # Analyze face for emotions, age, and gender
            analysis = DeepFace.analyze(face_crop, actions=['age', 'gender', 'emotion'], enforce_detection=False)

            detected_age = int(analysis[0]['age'])  # Convert to integer
            if stored_gender is None:
                stored_gender = analysis[0]['dominant_gender']

            # Stabilize age detection:  
            if stable_age is None:  
                stable_age = detected_age  # Store the first detected age
            else:
                # Keep the detected age within ±1 or ±2 years range
                if abs(detected_age - stable_age) > 2:
                    detected_age = stable_age  # Ignore sudden changes
                else:
                    stable_age = detected_age  # Update stable age slowly

            # Show exact two ages (e.g., 27-28)
            age_display = f"{stable_age}-{stable_age + 1}"

            dominant_emotion = analysis[0]['dominant_emotion']
            emotion_scores = analysis[0]['emotion']

            # Display detected attributes on the face
            cv2.putText(frame, f"{stored_gender}, Age: {age_display}", (x, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (x, y - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw vertical emotion chart next to the face
            chart_x = x + w + 30 if x + w + 250 < frame.shape[1] else x - 250  # Adjust position
            chart_y = y
            draw_vertical_emotion_chart(frame, emotion_scores, chart_x, chart_y)

        except Exception as e:
            print("DeepFace error:", e)

    # Display the webcam feed in fullscreen mode
    cv2.namedWindow('Face Detection with Age, Gender & Emotions', cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Face Detection with Age, Gender & Emotions', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow('Face Detection with Age, Gender & Emotions', frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
