import face_recognition
import os
import pickle

faces_folder = "faces"

known_encodings = []
known_ids = []
known_names = []

person_id_map = {}
current_id = 1

for f in os.listdir(faces_folder):
    if f.lower().endswith((".jpg", ".png")):
        path = os.path.join(faces_folder, f)

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if not encodings:
            print(f"❌ No face found in {f}")
            continue

        # Extract name from filename
        # Example: Elisee_1.jpg → Elisee
        name = os.path.splitext(f)[0].split("_")[0]

        # Assign same ID if name already exists
        if name not in person_id_map:
            person_id_map[name] = current_id
            current_id += 1

        person_id = person_id_map[name]

        # Store ALL encodings found in image
        for encoding in encodings:
            known_encodings.append(encoding)
            known_ids.append(person_id)
            known_names.append(name)

        print(f"✅ Loaded {f} as {name} (ID {person_id})")

# Save clean encoding file
with open("encodings.pkl", "wb") as f:
    pickle.dump({
        "encodings": known_encodings,
        "ids": known_ids,
        "names": known_names
    }, f)

print("\n🎯 Encodings saved successfully.")
print(f"Total persons: {len(person_id_map)}")
print(f"Total encodings: {len(known_encodings)}")