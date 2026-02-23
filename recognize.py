import cv2
import os
import pandas as pd
from datetime import datetime

# ---------------- ID → Name Mapping ----------------
id_name_map = {
    1: "Karna",
    2: "Ravi",
    3: "Suresh"
}

# ---------------- Load Recognizer ----------------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ---------------- Ensure CSV exists ----------------
if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w") as f:
        f.write("Name,Time,Date")

# ---------------- Excel Export ----------------
def export_to_excel():
    df = pd.read_csv("attendance.csv")
    df.to_excel("attendance.xlsx", index=False)

# ---------------- Attendance Logic ----------------
def markAttendance(name):
    today = datetime.now().strftime('%d-%m-%Y')

    with open("attendance.csv", "r+") as f:
        lines = f.readlines()
        for line in lines[1:]:
            entry = line.strip().split(',')
            if entry[0] == name and entry[2] == today:
                return

        time = datetime.now().strftime('%H:%M:%S')
        f.write(f"\n{name},{time},{today}")

    export_to_excel()

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("📷 Press 'q' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        faceROI = gray[y:y+h, x:x+w]

        # IMPORTANT: same preprocessing as training
        faceROI = cv2.resize(faceROI, (200, 200))
        faceROI = cv2.equalizeHist(faceROI)

        id, confidence = recognizer.predict(faceROI)

        # LBPH: lower confidence = better match
        if confidence < 75:
            name = id_name_map.get(id, "Unknown")
            markAttendance(name)
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            f"{name} ({int(confidence)})",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

    cv2.imshow("Smart Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
