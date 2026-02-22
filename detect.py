from flask import Flask, request, jsonify
from flask_cors import CORS  # ← import CORS
import face_recognition
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # ← enable CORS for all routes

# Load encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_ids = data["ids"]
known_names = data["names"]

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    image = face_recognition.load_image_file(file)
    
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    results = []
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        label = "Unknown"
        confidence = 0.0

        for i, match in enumerate(matches):
            if match:
                label = known_names[i]
                confidence = float(np.linalg.norm(face_encoding - known_encodings[i]))
                break

        results.append({
            "name": label,
            "confidence": confidence,
            "bbox": {"top": top, "right": right, "bottom": bottom, "left": left}
        })
    
    return jsonify({"faces": results})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)