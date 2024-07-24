#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Step 01
import cv2
import mediapipe as mp
import numpy as np

mp_draw = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# In[ ]:


# step 03

cap = cv2.VideoCapture(0)
# Setup Media pipe Instance
with mp_pose.Pose(min_detection_confidence=0., min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(25, 177, 66), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(25, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('WebCam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# In[ ]:


# Step 02

# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#    ret, frame = cap.read()
#    cv2.imshow('WebCam Feed', frame)

#    if cv2.waitKey(10) & 0xFF == ord('q'):
#        break
# cap.release()
# cv2.destroyAllWindows()


# In[ ]:


# Step 04
cap = cv2.VideoCapture(0)
# Setup Media pipe Instance
with mp_pose.Pose(min_detection_confidence=0., min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark
        except:
            pass

        # Render
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(25, 177, 66), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(25, 66, 230), thickness=2, circle_radius=2))
        image = cv2.resize(image, (640, 480))
        cv2.imshow('WebCam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


# In[ ]:


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle


# In[ ]:


shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

# In[ ]:


# In[ ]:


# Step 05
cap = cv2.VideoCapture(0)
# Setup Media pipe Instance
with mp_pose.Pose(min_detection_confidence=0., min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # get angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # dikhao
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [720, 520]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA
                        )


        except:
            pass

        # Render
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(25, 177, 66), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(25, 66, 230), thickness=2, circle_radius=2))
        image = cv2.resize(image, (720, 520))
        cv2.imshow('WebCam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# In[104]:


# Step 06 Counter
cap = cv2.VideoCapture(0)
counter = 0
stage = None

# Setup Media pipe Instance
with mp_pose.Pose(min_detection_confidence=0., min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        # Recolor Image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract Landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # get angle
            angle = calculate_angle(shoulder, elbow, wrist)

            # dikhao
            cv2.putText(image, str(angle),
                        tuple(np.multiply(elbow, [720, 520]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2, cv2.LINE_AA
                        )
            # Counter
            if angle > 160:
                stage = "Down"
            if angle < 45 and stage == "Down":
                stage = "Up"
                counter += 1


        except:
            pass

        # Draw the Box
        cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
        cv2.putText(image, "Reps", (15, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)

        cv2.putText(image, "Arm Stage", (65, 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, (60, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

        # Render
        mp_draw.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                               mp_draw.DrawingSpec(color=(25, 177, 66), thickness=2, circle_radius=3),
                               mp_draw.DrawingSpec(color=(25, 66, 230), thickness=2, circle_radius=2))
        image = cv2.resize(image, (720, 520))
        cv2.imshow('WebCam Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# In[ ]:


# In[ ]:




