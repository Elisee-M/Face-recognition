from flask import Flask, request, jsonify
from flask_cors import CORS
import face_recognition
import pickle
import numpy as np
import os
import uuid
import cv2

app = Flask(__name__)
CORS(app)

# ------------------------------
# Load existing encodings
# ------------------------------

ENCODING_FILE = "encodings.pkl"
FACES_FOLDER = "faces"

if os.path.exists(ENCODING_FILE):
    with open(ENCODING_FILE, "rb") as f:
        data = pickle.load(f)

    known_encodings = data["encodings"]
    known_ids = data["ids"]
    known_names = data["names"]

else:
    known_encodings = []
    known_ids = []
    known_names = []

# Recognition threshold
THRESHOLD = 0.5


# ------------------------------
# FACE DETECTION
# ------------------------------

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

        if len(known_encodings) == 0:

            label = "Unknown"
            confidence = 0
            new_person = True

        else:

            face_distances = face_recognition.face_distance(
                known_encodings, face_encoding
            )

            best_match_index = np.argmin(face_distances)
            best_distance = face_distances[best_match_index]

            print("Best distance:", best_distance)

            if best_distance < THRESHOLD:

                label = known_names[best_match_index]
                confidence = float(1 - best_distance)
                new_person = False

            else:

                label = "Unknown"
                confidence = 0
                new_person = True

        results.append({
            "name": label,
            "confidence": confidence,
            "new_person": new_person,
            "bbox": {
                "top": top,
                "right": right,
                "bottom": bottom,
                "left": left
            }
        })

    return jsonify({"faces": results})


# ------------------------------
# REGISTER NEW FACE
# ------------------------------

@app.route("/register", methods=["POST"])
def register():

    name = request.form.get("name")
    file = request.files["image"]

    if not name or not file:
        return jsonify({"error": "Missing name or image"}), 400

    image = face_recognition.load_image_file(file)

    encodings = face_recognition.face_encodings(image)

    if len(encodings) == 0:
        return jsonify({"error": "No face detected"}), 400

    encoding = encodings[0]

    # save image to dataset
    filename = f"{name}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(FACES_FOLDER, filename)

    cv2.imwrite(path, image[:, :, ::-1])

    # update memory
    person_id = len(set(known_ids)) + 1

    known_encodings.append(encoding)
    known_ids.append(person_id)
    known_names.append(name)

    # save new encoding file
    with open(ENCODING_FILE, "wb") as f:
        pickle.dump({
            "encodings": known_encodings,
            "ids": known_ids,
            "names": known_names
        }, f)

    return jsonify({
        "message": f"{name} registered successfully"
    })


# ------------------------------
# START SERVER
# ------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)