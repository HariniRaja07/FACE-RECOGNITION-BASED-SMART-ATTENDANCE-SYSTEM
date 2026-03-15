import streamlit as st
import cv2
import pandas as pd
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Face Recognition Attendance System ")

# ---------------------- Students ----------------------
# Map student names to their images
path = "students"
student_images = {}
for file in os.listdir(path):
    if file.endswith(".jpg") or file.endswith(".png"):
        name = os.path.splitext(file)[0].upper()
        img_path = os.path.join(path, file)
        student_images[name] = cv2.imread(img_path)

student_names = list(student_images.keys())

# ---------------------- Attendance Excel ----------------------
def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if os.path.exists("attendance.xlsx"):
        df = pd.read_excel("attendance.xlsx")
    else:
        df = pd.DataFrame(columns=["Name","Date","Time","Status"])

    records = df[df["Name"]==name]

    if records.empty or records.iloc[-1]["Status"]=="Exit":
        new_row = {"Name":name, "Date":date, "Time":time, "Status":"Entry"}
        st.success(f"Welcome {name}! 😊")
    else:
        new_row = {"Name":name, "Date":date, "Time":time, "Status":"Exit"}
        st.info(f"Goodbye {name}! 👋")

    df = pd.concat([df,pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel("attendance.xlsx", index=False)

# ---------------------- Face Detection ----------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # For simplicity, match by nearest student image size (or first student)
            # In real scenario, you can add template matching or other feature matching
            if student_names:
                name = student_names[0]  # Just pick first student for demo
                cv2.putText(img, name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                mark_attendance(name)

        return img

# ---------------------- Run Webcam ----------------------
webrtc_streamer(key="attendance", video_transformer_factory=VideoTransformer)
