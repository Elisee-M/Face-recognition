import face_recognition
import os

# -------------------------
# Step 1: Load Known Faces
# -------------------------

known_encodings = []
known_names = []

faces_folder = "faces"

for filename in os.listdir(faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        path = os.path.join(faces_folder, filename)

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)

print("Known faces loaded:", known_names)


# -------------------------
# Step 2: Test Recognition
# -------------------------

test_image_path = "test_images/five.jpg"  # change this to test others

test_image = face_recognition.load_image_file(test_image_path)
test_encodings = face_recognition.face_encodings(test_image)

if test_encodings:
    test_encoding = test_encodings[0]

    results = face_recognition.compare_faces(known_encodings, test_encoding)

    matched = False
    for i, match in enumerate(results):
        if match:
            print("Recognized as:", known_names[i])
            matched = True

    if not matched:
        print("Face not recognized.")

else:
    print("No face detected in test image.")
