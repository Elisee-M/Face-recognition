import face_recognition
import os
import pickle  # to save encodings for later use

# -------------------------
# Step 1: Load Known Faces and Save as Numbered IDs
# -------------------------

known_encodings = []
known_ids = []  # will store numbers instead of names

faces_folder = "faces"

photo_number = 1  # starting number

for filename in os.listdir(faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(faces_folder, filename)

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_ids.append(photo_number)
            print(f"Loaded {filename} as ID {photo_number}")
            photo_number += 1

print("All known faces loaded with IDs:", known_ids)

# Optionally save encodings for later so you don't have to reload
with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": known_encodings, "ids": known_ids}, f)

# -------------------------
# Step 2: Test Recognition
# -------------------------

test_image_path = "test_images/five.jpg"  # change this to test other images

test_image = face_recognition.load_image_file(test_image_path)
test_encodings = face_recognition.face_encodings(test_image)

if test_encodings:
    test_encoding = test_encodings[0]

    results = face_recognition.compare_faces(known_encodings, test_encoding)

    matched = False
    for i, match in enumerate(results):
        if match:
            print("Recognized as ID:", known_ids[i])
            matched = True

    if not matched:
        print("Face not recognized.")

else:
    print("No face detected in test image.")
