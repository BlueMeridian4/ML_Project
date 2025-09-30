import json

def save_keypoints(keypoints_list, out_file):
    with open(out_file, "w") as f:
        json.dump(keypoints_list, f)

def load_keypoints(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def get_bbox(landmarks, frame_shape):
    """Return bounding box (x1, y1, x2, y2) in pixel coords for pose landmarks."""
    h, w, _ = frame_shape
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)

def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return (x2 - x1) * (y2 - y1)

def compute_iou(bbox1, bbox2):
    """Compute IoU (intersection over union) between two bounding boxes."""
    x1, y1, x2, y2 = bbox1
    x1b, y1b, x2b, y2b = bbox2

    inter_x1 = max(x1, x1b)
    inter_y1 = max(y1, y1b)
    inter_x2 = min(x2, x2b)
    inter_y2 = min(y2, y2b)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)

    union_area = area1 + area2 - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

import numpy as np

# ----------------------------
# Smoothing / outlier helpers
# ----------------------------
def smooth_points(prev_points, new_points, alpha=0.75):
    """
    Exponential moving average for landmarks.
    prev_points, new_points: list of (x,y,z,visibility)
    """
    if prev_points is None:
        return new_points
    prev_arr = np.array(prev_points)
    new_arr = np.array(new_points)
    smoothed = alpha * prev_arr + (1 - alpha) * new_arr
    return smoothed.tolist()

def reject_outliers(prev_points, new_points, threshold=0.05):
    """
    Ignore sudden jumps in normalized coordinates (0-1 range).
    threshold: max allowed distance per joint
    """
    if prev_points is None:
        return new_points
    prev_arr = np.array(prev_points)
    new_arr = np.array(new_points)
    distances = np.linalg.norm(prev_arr[:, :3] - new_arr[:, :3], axis=1)
    for i, d in enumerate(distances):
        if d > threshold:
            new_arr[i] = prev_arr[i]  # keep previous point
    return new_arr.tolist()