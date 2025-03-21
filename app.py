from flask import Flask, request, jsonify
import cv2
import numpy as np

app = Flask(__name__)

@app.route('/detect', methods=['POST'])
def detect_face():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Load OpenCV face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    return jsonify({'faces_detected': len(faces)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
