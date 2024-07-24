import cv2
import dlib
import numpy as np
import streamlit as st
from imutils import face_utils
from scipy.spatial import distance
from pygame import mixer

# Initialize Pygame mixer for alert sound
mixer.init()
mixer.music.load("music.wav")

# Function to calculate the Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Set thresholds and parameters
thresh = 0.25
frame_check = 20

# Load face detection and facial landmark models
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

# Streamlit app
st.title("Drowsiness Detection")
run = st.checkbox('Run')
FRAME_WINDOW = st.image([])

# Initialize variables
flag = 0

# Open webcam video stream
cap = cv2.VideoCapture(0)

while run:
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to grab frame")
        break
    
    frame = cv2.resize(frame, (450, 300))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if gray is None or len(gray.shape) != 2:
        st.write("Gray image conversion failed")
        break
    
    
    subjects = detect(gray, 0)

    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 265),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not mixer.music.get_busy():
                    mixer.music.play()
        else:
            flag = 0

    # Display the frame in the Streamlit app
    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
