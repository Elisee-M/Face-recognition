import cv2
import face_recognition
import pickle
import numpy as np

# Load encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

THRESHOLD = 0.5

# Start camera
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small = small_frame[:, :, ::-1]

    face_locations = face_recognition.face_locations(rgb_small)
    face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        name = "Unknown"

        if len(known_encodings) > 0:
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match = np.argmin(distances)

            if distances[best_match] < THRESHOLD:
                name = known_names[best_match]

        # scale back up
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # draw box
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()