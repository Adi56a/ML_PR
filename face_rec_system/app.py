# app.py
from flask import Flask, render_template, request, jsonify
import face_recognition
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load reference image (registration phase assumed to be done already)
known_image = face_recognition.load_image_file("reference.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    data = request.get_json()
    image_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(image_data)

    np_arr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return jsonify({"message": "No face detected."})

    face_encodings = face_recognition.face_encodings(frame, face_locations)
    match = face_recognition.compare_faces([known_encoding], face_encodings[0])

    if match[0]:
        return jsonify({"message": "Face Verified ✅"})
    else:
        return jsonify({"message": "Face Not Recognized ❌"})

if __name__ == '__main__':
    app.run(debug=True)
