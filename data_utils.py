import json

def save_keypoints(keypoints_list, out_file):
    with open(out_file, "w") as f:
        json.dump(keypoints_list, f)

def load_keypoints(file_path):
    with open(file_path, "r") as f:
        return json.load(f)
