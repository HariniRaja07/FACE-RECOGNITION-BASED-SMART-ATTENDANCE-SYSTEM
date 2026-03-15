import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random
from streamlit_webrtc import webrtc_streamer

st.title("Face Recognition Attendance System")

# Motivational quotes
quotes = [
"Success starts with showing up today.",
"Every day is a new opportunity to learn.",
"Your future is created by what you do today.",
"Small progress is still progress.",
"Believe in yourself and keep going."
]

# Load student images
path = "students"
images = []
classNames = []

if os.path.exists(path):

    myList = os.listdir(path)

    for cl in myList:
        curImg = cv2.imread(f"{path}/{cl}")
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])

st.write("Start Camera")

# Face detection using OpenCV
def video_frame_callback(frame):

    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

        name = "Student"

        now = datetime.now()
        date = now.strftime("%Y-%m-%d")
        time = now.strftime("%H:%M:%S")

        if os.path.exists("attendance.xlsx"):
            df = pd.read_excel("attendance.xlsx")
        else:
            df = pd.DataFrame(columns=["Name","Date","Time","Status"])

        new_row = {"Name":name,"Date":date,"Time":time,"Status":"Entry"}
        df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)

        df.to_excel("attendance.xlsx",index=False)

        st.success(f"Welcome {name}! 😊")
        st.write("✨", random.choice(quotes))

    return img


webrtc_streamer(key="camera", video_frame_callback=video_frame_callback)

# Reset button
if st.button("Reset System"):
    st.rerun()
