import cv2
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

# -------- Face Recognizer --------
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.yml')

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# -------- ID to Name Mapping --------
id_name_map = {
    1: "Karna",
    2: "Ravi",
    3: "Suresh"
}

cap = None
running = False

# -------- Attendance Logic --------
def markAttendance(name):
    today = datetime.now().strftime('%d-%m-%Y')

    with open("attendance.csv", "r+") as f:
        lines = f.readlines()
        for line in lines:
            entry = line.strip().split(',')
            if entry[0] == name and entry[2] == today:
                return
        time = datetime.now().strftime('%H:%M:%S')
        f.write(f"\n{name},{time},{today}")

# -------- Camera Start --------
def start_camera():
    global cap, running
    if running:
        return

    cap = cv2.VideoCapture(0)
    running = True
    messagebox.showinfo("Info", "Attendance Started")

    while running:
        ret, img = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:
                name = id_name_map.get(id, "Unknown")
                markAttendance(name)
            else:
                name = "Unknown"

            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(img,name,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)

        cv2.imshow("Smart Attendance System", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_camera()

    cv2.destroyAllWindows()

# -------- Camera Stop --------
def stop_camera():
    global cap, running
    running = False
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Info", "Attendance Stopped")

# -------- GUI --------
root = tk.Tk()
root.title("Smart Attendance System")
root.geometry("300x200")

tk.Label(root, text="Smart Attendance System",
         font=("Arial", 14, "bold")).pack(pady=10)

tk.Button(root, text="Start Attendance",
          width=20, bg="green", fg="white",
          command=start_camera).pack(pady=10)

tk.Button(root, text="Stop Attendance",
          width=20, bg="red", fg="white",
          command=stop_camera).pack(pady=10)

root.mainloop()
print("GUI closed")