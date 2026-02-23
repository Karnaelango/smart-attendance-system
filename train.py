import cv2
import os
import numpy as np

# LBPH recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

# Face detector
detector = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def getImagesAndLabels(path):
    faceSamples = []
    ids = []

    for folder in os.listdir(path):
        folderPath = os.path.join(path, folder)

        if not os.path.isdir(folderPath):
            continue

        for image in os.listdir(folderPath):
            imgPath = os.path.join(folderPath, image)

            # Read image in grayscale
            img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Resize for consistency
            img = cv2.resize(img, (200, 200))

            # Improve lighting
            img = cv2.equalizeHist(img)

            # Detect face
            faces = detector.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(100, 100)
            )

            for (x, y, w, h) in faces:
                faceROI = img[y:y+h, x:x+w]
                faceSamples.append(faceROI)
                ids.append(int(folder))

    return faceSamples, ids
print("🔄 Training started...")

faces, ids = getImagesAndLabels('dataset')

if len(faces) == 0:
    print("❌ No faces found! Check dataset.")
    exit()

recognizer.train(faces, np.array(ids))
recognizer.save('trainer.yml')

print("✅ Training Completed Successfully")
print(f"📊 Total faces trained: {len(faces)}")
