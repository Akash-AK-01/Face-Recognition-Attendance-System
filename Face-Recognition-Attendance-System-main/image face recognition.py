import cv2
import os
import numpy as np
import csv
from datetime import datetime

# ---------- Paths ----------
DATASET_DIR = r"gpt/dataset"  # gpt/dataset/<StudentName>/*.jpg|png
ATTENDANCE_CSV = r"E:\Ak\Face-Recognition-Attendance-System-main\gpt\attendance.csv"
UNKNOWN_DIR = r"gpt/unknown"

os.makedirs(UNKNOWN_DIR, exist_ok=True)

# ---------- Helpers ----------
def load_dataset_and_train():
    """Load faces from gpt/dataset in memory and train LBPH (no .yml saved)."""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}
    current_label = 0
    images_count = 0

    # Walk each student's folder
    for person_name in sorted(os.listdir(DATASET_DIR)):
        person_dir = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_label] = person_name

        for filename in os.listdir(person_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue
            img_path = os.path.join(person_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # detect face in the training image (robust against extra background)
            dets = face_cascade.detectMultiScale(img, 1.3, 5)
            if len(dets) == 0:
                # if no face found, try using full image (some datasets are already cropped)
                roi = cv2.resize(img, (200, 200), interpolation=cv2.INTER_CUBIC)
                faces.append(roi)
                labels.append(current_label)
                images_count += 1
                continue

            for (x, y, w, h) in dets[:1]:  # take first face
                roi = img[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_CUBIC)
                faces.append(roi)
                labels.append(current_label)
                images_count += 1

        current_label += 1

    if images_count == 0:
        raise RuntimeError("No training images found. Put photos in gpt/dataset/<StudentName>/")

    recognizer.train(np.array(faces), np.array(labels))
    return recognizer, label_map

def ensure_attendance_csv():
    if not os.path.exists(ATTENDANCE_CSV):
        with open(ATTENDANCE_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "DateTime"])

def append_attendance_once(name, seen_set):
    """Mark attendance once per name for this session."""
    if name in seen_set:
        return
    seen_set.add(name)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(ATTENDANCE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, ts])
    print(f"📝 Marked present: {name} at {ts}")

# ---------- Train in memory ----------
print("🔄 Training LBPH in memory (no .yml file)…")
try:
    recognizer, label_map = load_dataset_and_train()
except Exception as e:
    print(f"❌ {e}")
    raise SystemExit(1)
print("✅ Training complete (in memory).")

# ---------- Live recognition ----------
ensure_attendance_csv()
face_cascade_live = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
seen_students = set()
unknown_saved = False  # save only one unknown snapshot per session

cap = cv2.VideoCapture(0)  # replace with RTSP/URL for CCTV if needed

CONFIDENCE_THRESHOLD = 60  # lower => stricter (fewer false positives)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade_live.detectMultiScale(gray, 1.3, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi_resized = cv2.resize(roi, (200, 200), interpolation=cv2.INTER_CUBIC)

        # Predict
        label, conf = recognizer.predict(roi_resized)

        if conf < CONFIDENCE_THRESHOLD and label in label_map:
            name = label_map[label]
            # draw & label (name only, no number)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 200, 0), 2)
            cv2.putText(frame, name, (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            append_attendance_once(name, seen_students)
            # if we saw a known student, allow unknown to be captured again later if desired:
            # (leave unknown_saved logic as-is to save only first unknown overall)
        else:
            # Unknown
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 220), 2)
            cv2.putText(frame, "Unknown", (x, y-12), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if not unknown_saved:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(UNKNOWN_DIR, f"unknown_{ts}.jpg")
                cv2.imwrite(save_path, roi_resized)
                unknown_saved = True
                print(f"⚠️ Unknown face saved once: {save_path}")

    cv2.imshow("In-Memory Face Attendance (OpenCV)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Attendance saved to: {ATTENDANCE_CSV}")
