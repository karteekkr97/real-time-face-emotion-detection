from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "Face Detection API is running!"

@app.route('/detect', methods=['POST'])
def detect_faces():
    # Get image from request
    file = request.files['image']
    image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Perform face detection (use your existing detection code)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Return results
    return jsonify({'faces_detected': len(faces)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Change port if needed
