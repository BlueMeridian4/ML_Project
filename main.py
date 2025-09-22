import cv2
from video_loader import load_video
from pose_estimator import PoseEstimator
from data_utils import save_keypoints
from visualize import draw_keypoints
import mediapipe as mp

def run_pipeline(video_path, out_json="keypoints.json"):
    pose_estimator = PoseEstimator()
    keypoints_all = []

    for frame in load_video(video_path):
        # Convert BGR (OpenCV) → RGB (MediaPipe requirement)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose_estimator.pose.process(frame_rgb)
        if results.pose_landmarks:
            keypoints = [(lm.x, lm.y, lm.z, lm.visibility) 
                         for lm in results.pose_landmarks.landmark]
            keypoints_all.append(keypoints)

            # Draw skeleton overlay
            frame_out = draw_keypoints(frame, results)
            cv2.imshow("Pose Detection", frame_out)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    save_keypoints(keypoints_all, out_json)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline("bjj_role.mp4", out_json="bjj_keypoints.json")
