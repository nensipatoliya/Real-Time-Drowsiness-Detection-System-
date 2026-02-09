#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import dlib
import numpy as np
import winsound  # For alarm sound on Windows
from scipy.spatial import distance as dist
import time

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    alarm_on = False
    closed_eye_frames = 0  # To count how many frames eyes are closed
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get frame rate of the camera
    required_frames = int(frame_rate * 2)  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = detector(gray)

        for face in faces:
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)

            # Get facial landmarks
            landmarks = predictor(gray, face)

            # Extract left and right eye landmarks
            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            # Compute EAR
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            # Drowsiness detection threshold
            status = "Awake"
            color = (0, 255, 0)  # Green (awake)

            if avg_EAR < 0.25:
                closed_eye_frames += 1
                if closed_eye_frames >= required_frames:
                    status = "Drowsy"
                    color = (0, 0, 255)  # Red (drowsy)
                    if not alarm_on:
                        winsound.Beep(1000, 2000)  # Sound alarm for 2 seconds
                        alarm_on = True
            else:
                closed_eye_frames = 0  # Reset counter if eyes are open
                alarm_on = False  # Reset alarm state

            # Display status
            cv2.putText(frame, f"Status: {status}", (face.left(), face.top() - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display the frame
        cv2.imshow("Drowsiness Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()


# In[3]:


import cv2
import mediapipe as mp
import numpy as np
import winsound  # Windows only (for alarm)
from scipy.spatial import distance as dist

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Function to calculate Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Indices for left and right eye landmarks in Mediapipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]  # Adjusted for Mediapipe
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Function to detect eyes when landmarks fail
def detect_eyes_fallback(gray):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return eyes

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    alarm_on = False
    closed_eye_frames = 0
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    required_frames = int(frame_rate * 2)  # 2 seconds of closed eyes
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to grayscale and enhance contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # Improve visibility in low light
        
        # Convert frame to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(rgb_frame)
        eye_detected = False
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract left and right eye landmarks
                left_eye = np.array([(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                      int(face_landmarks.landmark[i].y * frame.shape[0])) for i in LEFT_EYE])
                right_eye = np.array([(int(face_landmarks.landmark[i].x * frame.shape[1]),
                                       int(face_landmarks.landmark[i].y * frame.shape[0])) for i in RIGHT_EYE])
                
                # Compute EAR
                left_EAR = eye_aspect_ratio(left_eye)
                right_EAR = eye_aspect_ratio(right_eye)
                avg_EAR = (left_EAR + right_EAR) / 2.0
                
                if avg_EAR > 0:
                    eye_detected = True
                
                # Drowsiness detection threshold
                status = "Awake"
                color = (0, 255, 0)  # Green (awake)
                
                if avg_EAR < 0.25:
                    closed_eye_frames += 1
                    if closed_eye_frames >= required_frames:
                        status = "Drowsy"
                        color = (0, 0, 255)  # Red (drowsy)
                        if not alarm_on:
                            winsound.Beep(1000, 2000)  # Alarm for 2 seconds
                            alarm_on = True
                else:
                    closed_eye_frames = 0  # Reset counter
                    alarm_on = False  # Reset alarm state
                
                # Draw eye landmarks
                for point in left_eye:
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)
                for point in right_eye:
                    cv2.circle(frame, point, 2, (255, 0, 0), -1)
                
                # Display status
                cv2.putText(frame, f"Status: {status}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Fallback to eye detection if Mediapipe fails (e.g., black glasses)
        if not eye_detected:
            eyes = detect_eyes_fallback(gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                status = "Awake"
                closed_eye_frames = 0  # Reset counter
                alarm_on = False  # Reset alarm state
                cv2.putText(frame, f"Status: {status}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Drowsiness Detection", frame)
        
        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()

