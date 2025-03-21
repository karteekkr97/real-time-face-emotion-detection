# Real-Time Face & Emotion Detection

## About the Project
This project is a real-time face and emotion detection system using OpenCV and TensorFlow. It leverages computer vision and deep learning to detect human faces and classify their emotions (e.g., happy, sad, angry, neutral) from live camera feed or video input.

## Features
- Real-time face detection using OpenCV.
- Emotion recognition with deep learning models.
- Support for video files and live webcam feed.
- Easy installation and usage.

## Installation
### Prerequisites
Ensure you have the following installed before proceeding:
- Python 3.7 or later
- Git
- Virtual environment (recommended)

### Steps to Install and Run
1. **Clone the Repository**
   ```sh
   git clone https://github.com/karteekkr97/real-time-face-emotion-detection.git
   cd real-time-face-emotion-detection
   ```

2. **Create and Activate Virtual Environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```sh
   python face_detection.py
   ```

## Usage
- The script will access your webcam by default.
- Detected faces will be highlighted in real-time.
- Emotion classification results will be displayed on the video feed.

## Technologies Used
- **Python**: Main programming language.
- **OpenCV**: For face detection.
- **TensorFlow/Keras**: For deep learning-based emotion classification.

## Contributing
Feel free to fork the repository and submit pull requests for improvements or additional features.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Author
Developed by Karteekkr. If you have any questions, reach out via GitHub!

