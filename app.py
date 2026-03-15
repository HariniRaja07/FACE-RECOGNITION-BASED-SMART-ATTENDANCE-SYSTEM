from streamlit_webrtc import webrtc_streamer
import streamlit as st
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
import random

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

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Encode faces
def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)

        if len(encodes) > 0:
            encodeList.append(encodes[0])

    return encodeList


encodeListKnown = findEncodings(images)

st.write("Click the checkbox to start camera")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

if run:

    cap = cv2.VideoCapture(0)

    while True:

        success, img = cap.read()

        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):

            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:

                name = classNames[matchIndex].upper()

                # Draw face box
                y1,x2,y2,x1 = faceLoc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4

                cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(img,name,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2)

                now = datetime.now()
                date = now.strftime("%Y-%m-%d")
                time = now.strftime("%H:%M:%S")

                if os.path.exists("attendance.xlsx"):
                    df = pd.read_excel("attendance.xlsx")
                else:
                    df = pd.DataFrame(columns=["Name","Date","Time","Status"])

                person_records = df[df["Name"]==name]

                # ENTRY
                if person_records.empty or person_records.iloc[-1]["Status"]=="Exit":

                    new_row = {"Name":name,"Date":date,"Time":time,"Status":"Entry"}
                    df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)

                    st.success(f"Welcome {name}! 😊")
                    st.write("✨", random.choice(quotes))

                # EXIT
                else:

                    new_row = {"Name":name,"Date":date,"Time":time,"Status":"Exit"}
                    df = pd.concat([df,pd.DataFrame([new_row])],ignore_index=True)

                    st.info(f"Goodbye {name}! 👋")
                    st.write("💡", random.choice(quotes))

                df.to_excel("attendance.xlsx",index=False)

                FRAME_WINDOW.image(img, channels="BGR")

                cap.release()
                cv2.destroyAllWindows()

                # RESET BUTTON
                if st.button("Reset System"):
                    st.rerun()

                st.stop()

        FRAME_WINDOW.image(img, channels="BGR")

import cv2
from streamlit_webrtc import webrtc_streamer

st.write("Start Camera")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

    return img

webrtc_streamer(key="camera", video_frame_callback=video_frame_callback)
