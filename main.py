import face_recognition
import os
import pickle

faces_folder = "faces"

known_encodings = []
known_ids = []
known_names = []

photo_number = 1

for f in os.listdir(faces_folder):
    if f.endswith((".jpg", ".png")):
        path = os.path.join(faces_folder, f)

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_ids.append(photo_number)

            # filename without extension = name
            name = os.path.splitext(f)[0]
            known_names.append(name)

            print(f"Loaded {name} as ID {photo_number}")
            photo_number += 1

pickle.dump({
    "encodings": known_encodings,
    "ids": known_ids,
    "names": known_names
}, open("encodings.pkl", "wb"))

print("Encodings saved with IDs and names.")
