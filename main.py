import face_recognition
import os
import pickle
import cv2

# -------------------------
# FOLDERS
# -------------------------
faces_folder = "faces"
drawn_folder = "drawn_faces"

os.makedirs(drawn_folder, exist_ok=True)

known_encodings = []
known_ids = []
photo_number = 1

# -------------------------
# STEP 1: LOAD + DRAW LANDMARKS
# -------------------------

for f in os.listdir(faces_folder):
    if f.endswith((".jpg", ".png")):
        path = os.path.join(faces_folder, f)

        # Load image
        image = face_recognition.load_image_file(path)

        # Convert for OpenCV (RGB → BGR)
        image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Get encodings
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_ids.append(photo_number)

            # Get facial landmarks
            landmarks_list = face_recognition.face_landmarks(image)

            # Draw landmarks
            for landmarks in landmarks_list:
                for feature in landmarks.values():
                    for point in feature:
                        cv2.circle(image_cv, point, 2, (0, 255, 255), -1)

            # Save drawn image
            save_path = os.path.join(drawn_folder, f)
            cv2.imwrite(save_path, image_cv)

            print(f"Processed {f} → ID {photo_number}")
            photo_number += 1

# Save encodings
pickle.dump({"encodings": known_encodings, "ids": known_ids},
            open("encodings.pkl", "wb"))

print("Done. Faces encoded + patterns drawn.")
