import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
import pandas as pd
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("Face Recognition Attendance System")

# Load known faces
path = "faces"
images = []
classNames = []

myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f"{path}/{cl}")
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = findEncodings(images)


def markAttendance(name):
    file = "Attendance.csv"

    if not os.path.exists(file):
        with open(file, "w") as f:
            f.write("Name,Time\n")

    with open(file, "r+") as f:
        data = f.readlines()
        nameList = []

        for line in data:
            entry = line.split(",")
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            time = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name},{time}")


class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()

                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                markAttendance(name)

        return img


st.subheader("Start Camera")

webrtc_streamer(key="attendance", video_transformer_factory=VideoTransformer)

if os.path.exists("Attendance.csv"):
    df = pd.read_csv("Attendance.csv")
    st.subheader("Attendance Record")
    st.dataframe(df)
