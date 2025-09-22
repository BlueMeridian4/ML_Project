import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils

def draw_keypoints(frame, results):
    if results and results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
    return frame
