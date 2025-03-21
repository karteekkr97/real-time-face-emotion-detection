import os
from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

# Load OpenCV face detection model
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

if face_cascade.empty():
    raise ValueError("Error loading Haarcascade XML. Make sure OpenCV is installed correctly.")

@app.route('/detect', methods=['POST'])
def detect_face():
    file = request.files.get('image')  # Get uploaded image
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Convert image to numpy array
        npimg = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({'error': 'Invalid image format'}), 400

        # Detect faces
        faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)

        return jsonify({'faces_detected': len(faces) if isinstance(faces, np.ndarray) else 0})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's dynamic port
    app.run(host='0.0.0.0', port=port, debug=True)
