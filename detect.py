import face_recognition
import pickle
import cv2

# Load encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_ids = data["ids"]
known_names = data["names"]

# Load test image
image = face_recognition.load_image_file("test_images/mm.jpg")
image_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

    matches = face_recognition.compare_faces(known_encodings, face_encoding)
    label = "Unknown"

    for i, match in enumerate(matches):
        if match:
            label = f"ID {known_ids[i]} - {known_names[i]}"
            print("Recognized:", label)
            break

    cv2.rectangle(image_cv, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image_cv, label, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

cv2.imshow("Detection", image_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
