import mediapipe as mp

mp_pose = mp.solutions.pose

class PoseEstimator:
    def __init__(self, static_image_mode=False, min_detection_confidence=0.5):
        self.pose = mp_pose.Pose(
            static_image_mode=static_image_mode,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )

    def detect(self, frame):
        results = self.pose.process(frame)
        if not results.pose_landmarks:
            return None
        # Extract joints as (x,y,z,visibility)
        keypoints = [(lm.x, lm.y, lm.z, lm.visibility) 
                     for lm in results.pose_landmarks.landmark]
        return keypoints
