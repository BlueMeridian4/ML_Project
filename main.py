import cv2
from video_loader import load_video
from pose_estimator import PoseEstimator
from data_utils import save_keypoints, get_bbox, compute_iou, smooth_points, reject_outliers
from visualize import draw_keypoints
import mediapipe as mp

def run_pipeline(video_path, out_json="keypoints.json"):
    pose_estimator = PoseEstimator()
    keypoints_all = []
    tracked_bbox = None
    prev_keypoints = None  # for smoothing

    for frame in load_video(video_path):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose_estimator.pose.process(frame_rgb)

        if results.pose_landmarks:
            bbox = get_bbox(results.pose_landmarks.landmark, frame.shape)

            # Initialize tracking on first frame
            if tracked_bbox is None:
                tracked_bbox = bbox

            # Compare with last tracked bbox
            iou = compute_iou(bbox, tracked_bbox)
            if iou < 0.2:
                continue  # skip likely partner
            tracked_bbox = bbox

            # Extract keypoints
            keypoints = [(lm.x, lm.y, lm.z, lm.visibility) 
                         for lm in results.pose_landmarks.landmark]

            # Reject outliers first
            keypoints = reject_outliers(prev_keypoints, keypoints, threshold=0.05)

            # Smooth the keypoints
            keypoints = smooth_points(prev_keypoints, keypoints, alpha=0.75)
            prev_keypoints = keypoints  # update for next frame

            keypoints_all.append(keypoints)

            # Draw
            frame_out = draw_keypoints(frame, results)
            cv2.rectangle(frame_out, (int(bbox[0]), int(bbox[1])), 
                          (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.imshow("Pose Detection (Tracked & Smoothed)", frame_out)

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    save_keypoints(keypoints_all, out_json)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_pipeline("bjj_role.mp4", out_json="bjj_keypoints.json")