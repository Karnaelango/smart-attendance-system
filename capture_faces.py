import cv2
import os

# -------- USER SETTINGS --------
person_id = input("Enter person ID(number): ")          # 👈 Change ID for each person
person_name = input("Enter person name: ")  # Just for display
save_count = int(input("Enter number of images to capture: "))          # Number of images to capture

# -------- Create folder --------
dataset_path = f"dataset/{person_id}"
os.makedirs(dataset_path, exist_ok=True)

# -------- Face detector --------
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)
count = 0

print("📸 Dataset capture started...")
print("➡ Look straight, left, right, smile slightly")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        faceROI = gray[y:y+h, x:x+w]
        faceROI = cv2.resize(faceROI, (200, 200))
        faceROI = cv2.equalizeHist(faceROI)

        count += 1
        cv2.imwrite(f"{dataset_path}/{count}.jpg", faceROI)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
        cv2.putText(
            frame,
            f"{person_name} - {count}",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,255,255),
            2
        )

        cv2.waitKey(200)  # slight delay between captures

    cv2.imshow("Dataset Capture", frame)

    if count >= save_count:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ Dataset capture completed")
print(f"👤 Captured {count} images for ID: {person_id}, Name: {person_name}"    )