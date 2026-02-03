import cv2
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm

class VideoProcessor:
    """
    Handles extraction of features from video files including:
    - Keyframe extraction (first and middle frames)
    - Motion analysis (temporal pixel variations)
    - Scene complexity (variance of Laplacian across frames)
    """
    def __init__(self, device: str = "cpu"):
        self.device = device

    def process_batch(self, video_paths: list[str]) -> list[dict]:
        results = []
        for path in tqdm(video_paths, desc="Processing videos"):
            if not path.endswith((".mp4", ".avi", ".mov", ".mkv")):
                results.append(self._get_empty_features())
                continue

            results.append(self.process_video(path))
        return results

    def process_video(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return self._get_empty_features()

        frames = []
        count = 0
        # Sample up to 50 frames to keep it efficient
        while count < 50:
            ret, frame = cap.read()
            if not ret:
                break
            # Sample every 2nd frame if video is long? No, let's just take first 50
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            count += 1
        cap.release()

        if not frames:
            return self._get_empty_features()

        # 1. Motion Analysis
        # Calculate mean absolute difference between consecutive sampled frames
        motion_diffs = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float)))
            motion_diffs.append(diff)

        avg_motion = np.mean(motion_diffs) if motion_diffs else 0.0
        max_motion = np.max(motion_diffs) if motion_diffs else 0.0

        # 2. Keyframes
        # We take the middle frame as the representative keyframe
        middle_idx = len(frames) // 2
        keyframe = Image.fromarray(frames[middle_idx])

        # 3. Scene Complexity (Edge density variation)
        # Using Laplacian variance as a proxy for sharpness/complexity
        complexities = [cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var() for f in frames]
        avg_complexity = np.mean(complexities)

        return {
            "avg_motion": float(avg_motion / 255.0),
            "max_motion": float(max_motion / 255.0),
            "avg_complexity": float(avg_complexity / 1000.0), # Normalized roughly
            "num_frames": len(frames),
            "keyframe": keyframe
        }

    def _get_empty_features(self) -> dict:
        return {
            "avg_motion": 0.0,
            "max_motion": 0.0,
            "avg_complexity": 0.0,
            "num_frames": 0,
            "keyframe": None
        }
