"""
narrative_convergence.py

This script implements a full pipeline for computing a single stereoscopic
convergence value per video based on narrative importance. The code reads
RGB videos alongside aligned depth videos or image sequences, runs object
detection and tracking, extracts a variety of features per tracked object
(faces, motion, saliency, duration, composition), combines them into a unified
importance score using a context-aware dynamic weighting system, extracts the
corresponding median depth from the depth frames, smooths the resulting depth
signal, and writes a JSON file for each depth map containing its convergence
value.

This version has been heavily refactored for maximum performance with a
CUDA-enabled OpenCV build. It utilizes a full GPU pipeline to minimize
CPU-GPU data transfers.

Usage (example)::

python narrative_convergence.py
--videos /path/to/rgb_videos
--depths /path/to/depth_videos
--tracker deepsort --detector yolo12s --gaze auto
--window 9 --device cuda --verbose

"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
from tqdm import tqdm

# Try importing PyTorch.  Torch is required for the detectors and may
# automatically select CUDA if available.
try:
    import torch
except ImportError:
    torch = None

# Try importing ultralytics for YOLO models.
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None

# Try importing the deep_sort_realtime package.  If unavailable we fall
# back to a simple tracker.
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort  # type: ignore
    HAVE_DEEPSORT = True
except Exception:
    HAVE_DEEPSORT = False

# Try importing SciPy for the Kalman filter.  If unavailable we will
# implement a tiny Kalman filter ourselves.
try:
    from scipy.linalg import block_diag  # type: ignore
except Exception:
    block_diag = None

###########################################################################
# Utility functions and classes
###########################################################################

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists."""
    path.mkdir(parents=True, exist_ok=True)


def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union of two bounding boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    inter = interW * interH
    areaA = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    areaB = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    denom = areaA + areaB - inter
    return (inter / denom) if denom > 0 else 0.0


def resize_and_pad(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Resize an image while maintaining aspect ratio and pad to target size."""
    h, w = img.shape[:2]
    target_w, target_h = size
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    return padded


###########################################################################
# Model Registry
###########################################################################

class ModelRegistry:
    """Manage model files: download if missing, keep in ./models."""
    def __init__(self, root: Path) -> None:
        self.root = root
        ensure_dir(self.root)

    def get(self, filename: str, urls: List[str]) -> Optional[Path]:
        """Retrieve a model file."""
        path = self.root / filename
        if path.exists():
            return path
        for url in urls:
            try:
                import urllib.request
                print(f"Downloading {filename} from {url}...", file=sys.stderr)
                urllib.request.urlretrieve(url, str(path))
                return path
            except Exception as e:
                print(f"Failed to download {url}: {e}", file=sys.stderr)
                continue
        warnings.warn(f"Could not download model file {filename}")
        return None


###########################################################################
# Detectors
###########################################################################

class YOLODetector:
    """Wrapper around the Ultralytics YOLO detector."""
    def __init__(self, device: str = "cpu", model_name: str = "yolo12s") -> None:
        if YOLO is None:
            raise RuntimeError("Ultralytics YOLO is not available")
        self.device = device
        try:
            self.model = YOLO(f"{model_name}.pt")
        except Exception as e:
            warnings.warn(f"Failed to load model {model_name}.pt: {e}")
            raise RuntimeError(f"Could not load YOLO model {model_name}")
        if device == "cuda" and torch is not None and torch.cuda.is_available():
            self.model.to("cuda")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """Run detection on a frame."""
        results = self.model.predict(frame, verbose=False)
        detections: List[Tuple[int, int, int, int, float, int]] = []
        for r in results:
            boxes = r.boxes if hasattr(r, "boxes") else r
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                conf = float(b.conf[0].cpu().numpy())
                cls_id = int(b.cls[0].cpu().numpy()) if hasattr(b, "cls") else 0
                detections.append((int(x1), int(y1), int(x2), int(y2), conf, cls_id))
        return detections


###########################################################################
# Simple Tracker (fallback)
###########################################################################

@dataclass
class SimpleTrack:
    """Data class to hold state for a single tracked object."""
    id: int
    bbox: Tuple[int, int, int, int]
    age: int = 0
    last_seen: int = 0
    cls: int = 0
    prev_depth: float = 0.5
    median_depth: float = 0.5


class SimpleTracker:
    """A naive IoU‐based multi–object tracker."""
    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.next_id = 0
        self.tracks: Dict[int, SimpleTrack] = {}
        self.iou_threshold = iou_threshold

    def update(self, detections: List[Tuple[int, int, int, int, float, int]]) -> List[SimpleTrack]:
        updated_tracks: Dict[int, SimpleTrack] = {}
        assigned = set()
        for tid, track in self.tracks.items():
            best_iou, best_det_idx = 0.0, None
            for i, det in enumerate(detections):
                if i in assigned: continue
                score = iou(track.bbox, det[:4])
                if score > best_iou:
                    best_iou, best_det_idx = score, i
            if best_det_idx is not None and best_iou >= self.iou_threshold:
                x1, y1, x2, y2, _, cls = detections[best_det_idx]
                updated_tracks[tid] = SimpleTrack(id=tid, bbox=(x1, y1, x2, y2), age=track.age + 1, last_seen=0, cls=cls, prev_depth=track.median_depth)
                assigned.add(best_det_idx)
        for i, det in enumerate(detections):
            if i not in assigned:
                x1, y1, x2, y2, _, cls = det
                updated_tracks[self.next_id] = SimpleTrack(id=self.next_id, bbox=(x1, y1, x2, y2), age=1, last_seen=0, cls=cls)
                self.next_id += 1
        for tid, track in self.tracks.items():
            if tid not in updated_tracks:
                track.last_seen += 1
                if track.last_seen <= 2: updated_tracks[tid] = track
        self.tracks = updated_tracks
        return list(updated_tracks.values())


###########################################################################
# Face Detector
###########################################################################

class FaceDetector:
    """Wrapper around OpenCV's DNN face detector."""
    def __init__(self, registry: ModelRegistry, device: str = "cpu") -> None:
        self.model_loaded = False
        prototxt_name = "deploy.prototxt.txt"
        caffemodel_name = "res10_300x300_ssd_iter_140000.caffemodel"
        prototxt_urls = ["https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt.txt"]
        caffemodel_urls = ["https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"]
        self.prototxt_path = registry.get(prototxt_name, prototxt_urls)
        self.caffemodel_path = registry.get(caffemodel_name, caffemodel_urls)
        if self.prototxt_path and self.caffemodel_path:
            try:
                self.net = cv2.dnn.readNetFromCaffe(str(self.prototxt_path), str(self.caffemodel_path))
                if device == "cuda" and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                    print("Attempting to set OpenCV DNN backend to CUDA for FaceDetector.")
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                self.model_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load face detector: {e}")
        else:
            warnings.warn("Face detector model files are missing; face scores will be zero.")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame."""
        if not self.model_loaded: return []
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes: List[Tuple[int, int, int, int]] = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                boxes.append((max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)))
        return boxes

    def score_faces(self, frame: np.ndarray, tracks: List[SimpleTrack]) -> List[float]:
        """Compute continuous face scores for each track based on face size."""
        if not tracks: return []
        if not self.model_loaded: return [0.0] * len(tracks)
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        faces = self.detect_faces(frame)
        scores = []
        for track in tracks:
            max_face_score = 0.0
            for face in faces:
                if iou(track.bbox, face) > 0.1:
                    face_w, face_h = face[2] - face[0], face[3] - face[1]
                    score = (face_w * face_h) / frame_area
                    if score > max_face_score: max_face_score = score
            scores.append(max_face_score)
        return scores


###########################################################################
# Saliency (Gaze) Models
###########################################################################

class SaliencyModel:
    """Compute a saliency (gaze) heatmap for a frame."""
    def __init__(self, frame_size: Tuple[int, int]) -> None:
        self.frame_size = frame_size

    def compute_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Return a saliency heatmap in [0,1] for the input frame."""
        h, w = self.frame_size
        y = np.arange(h) - (h - 1) / 2.0
        x = np.arange(w) - (w - 1) / 2.0
        xx, yy = np.meshgrid(x, y)
        sigma_x, sigma_y = w / 6.0, h / 6.0
        heatmap = np.exp(-((xx**2) / (2 * sigma_x**2) + (yy**2) / (2 * sigma_y**2)))
        heatmap = heatmap.astype(np.float32)
        heatmap -= heatmap.min()
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap

    def score(self, tracks: List[SimpleTrack], heatmap: np.ndarray) -> List[float]:
        """Compute saliency scores for each track given a heatmap."""
        scores: List[float] = []
        h, w = heatmap.shape[:2]
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
            x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))
            region = heatmap[y1 : y2 + 1, x1 : x2 + 1]
            scores.append(float(region.mean()) if region.size > 0 else 0.0)
        return scores


###########################################################################
# Smoothing
###########################################################################

class MovingAverageSmoother:
    """Compute a moving average over a sequence."""
    def __init__(self, window: int = 9) -> None:
        self.window = max(1, window)

    def smooth(self, values: List[float]) -> List[float]:
        if not values: return []
        kernel = np.ones(self.window) / self.window
        padded = np.pad(values, (self.window // 2, self.window - 1 - self.window // 2), mode="edge")
        return np.convolve(padded, kernel, mode="valid").tolist()


class KalmanSmoother:
    """A simple 1D Kalman filter for smoothing depth signals."""
    def __init__(self, process_var: float = 1e-4, measurement_var: float = 1e-2) -> None:
        self.x = np.array([[0.5], [0.0]], dtype=np.float64)
        self.P = np.eye(2)
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        self.Q = np.array([[process_var, 0.0], [0.0, process_var]], dtype=np.float64)
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        self.R = np.array([[measurement_var]], dtype=np.float64)

    def smooth(self, values: List[float]) -> List[float]:
        if not values: return []
        smoothed: List[float] = []
        for z in values:
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            y = np.array([[z]], dtype=np.float64) - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            self.P = (np.eye(2) - K @ self.H) @ self.P
            smoothed.append(float(self.x[0, 0]))
        return smoothed


###########################################################################
# Narrative Processor
###########################################################################

class NarrativeProcessor:
    """Main orchestrator for computing narrative convergence values."""
    def __init__(self, video_dir: Path, depth_dir: Path, **kwargs) -> None:
        self.video_dir = video_dir
        self.depth_dir = depth_dir
        # Set attributes from kwargs with defaults
        self.detector_name = kwargs.get("detector_name", "yolo12s")
        self.tracker_name = kwargs.get("tracker_name", "deepsort")
        self.device = kwargs.get("device", "cpu")
        self.max_videos = kwargs.get("max_videos")
        self.verbose = kwargs.get("verbose", False)
        self.stride = max(1, int(kwargs.get("stride", 1)))
        self.debug_video = bool(kwargs.get("debug_video", False))
        self.debug_out = kwargs.get("debug_out")
        self.debug_scale = float(kwargs.get("debug_scale", 1.0))
        self.debug_layers = {layer.strip().lower() for layer in kwargs.get("debug_layers", []) if layer.strip()} if kwargs.get("debug_layers") else None
        
        self.logger = kwargs.get("logger") or (tqdm.write if self.verbose else (lambda _: None))
        set_seed(kwargs.get("seed", 42))
        self.registry = ModelRegistry(Path("models"))
        self.detector = self._init_detector()
        self.tracker = self._init_tracker()
        self.face_detector = FaceDetector(self.registry, device=self.device)
        self.smoother = KalmanSmoother() if kwargs.get("use_kalman") else MovingAverageSmoother(kwargs.get("window", 9))

    def _init_detector(self) -> Any:
        if self.detector_name.lower() in ["yolo12n", "yolo12s", "yolo12m"] and YOLO:
            return YOLODetector(device=self.device, model_name=self.detector_name)
        warnings.warn(f"Unsupported detector '{self.detector_name}' or Ultralytics not available.")
        return None

    def _init_tracker(self) -> Any:
        if self.tracker_name.lower() == "deepsort" and HAVE_DEEPSORT:
            return DeepSort(max_age=30)
        warnings.warn(f"Tracker '{self.tracker_name}' not 'deepsort' or DeepSORT unavailable; falling back to simple tracker.")
        return SimpleTracker()

    def _get_video_pairs(self) -> List[Tuple[Path, Path]]:
        def base_from_depth_name(p: Path) -> str:
            stem = p.stem
            return stem[:-6] if stem.endswith("_depth") else stem
        video_files = sorted([p for p in self.video_dir.iterdir() if p.is_file() or p.is_dir()])
        depth_files = {base_from_depth_name(p): p for p in self.depth_dir.iterdir()}
        return [(vf, depth_files[vf.stem]) for vf in video_files if vf.stem in depth_files]

    def _score_composition(self, tracks: List[SimpleTrack], frame_size: Tuple[int, int]) -> List[float]:
        h, w = frame_size
        thirds_x, thirds_y = [w / 3, 2 * w / 3], [h / 3, 2 * h / 3]
        scores = []
        for track in tracks:
            cx, cy = (track.bbox[0] + track.bbox[2]) / 2, (track.bbox[1] + track.bbox[3]) / 2
            norm_dist_center = math.sqrt(((abs(cx - w / 2) / (w / 2))**2 + (abs(cy - h / 2) / (h / 2))**2))
            center_score = math.exp(-2.0 * norm_dist_center**2)
            min_dist_thirds_x = min(abs(cx - thirds_x[0]), abs(cx - thirds_x[1]))
            min_dist_thirds_y = min(abs(cy - thirds_y[0]), abs(cy - thirds_y[1]))
            norm_dist_thirds = math.sqrt(((min_dist_thirds_x / (w / 6))**2 + (min_dist_thirds_y / (h / 6))**2))
            thirds_score = math.exp(-0.5 * norm_dist_thirds**2)
            scores.append(max(center_score, thirds_score))
        return scores

    def _compute_final_scores(self, scores: Dict[str, List[float]], global_motion: float, has_large_face: bool) -> Tuple[List[float], str]:
        num_tracks = len(scores.get("face", []))
        if num_tracks == 0: return [], "standard"
        context = "action" if global_motion > 5.0 else ("dialogue" if has_large_face and global_motion < 2.0 else "standard")
        weights = {
            "standard": {"face": 4.0, "gaze": 2.0, "motion": 1.5, "duration": 1.0, "composition": 1.0},
            "action":   {"face": 2.0, "gaze": 1.0, "motion": 4.0, "duration": 1.0, "composition": 0.5},
            "dialogue": {"face": 5.0, "gaze": 2.5, "motion": 0.5, "duration": 1.0, "composition": 1.5},
        }[context]
        norm_scores = {}
        for key, values in scores.items():
            arr = np.array(values, dtype=np.float32)
            min_v, max_v = arr.min(), arr.max()
            norm_scores[key] = (arr - min_v) / (max_v - min_v) if max_v > min_v else np.zeros_like(arr)
        base = (norm_scores.get("face", 0) * weights["face"] + norm_scores.get("gaze", 0) * weights["gaze"] + norm_scores.get("motion", 0) * weights["motion"])
        final = base * (1.0 + norm_scores.get("duration", 0) * weights["duration"]) * (1.0 + norm_scores.get("composition", 0) * weights["composition"])
        return final.tolist(), context

    def process(self) -> None:
        pairs = self._get_video_pairs()
        if self.max_videos is not None: pairs = pairs[:self.max_videos]
        master_list = {}
        iterator = tqdm(pairs, desc="Videos", disable=not self.verbose)
        for rgb_path, depth_path in iterator:
            if self.verbose: iterator.set_description(f"Processing {rgb_path.name}")
            try:
                conv = self._process_single(rgb_path, depth_path)
                master_list[rgb_path.stem] = conv
                with open(depth_path.with_suffix('.json'), "w", encoding="utf-8") as f:
                    json.dump({"convergence_plane": conv}, f, indent=4)
                self.logger(f"Saved convergence to {depth_path.with_suffix('.json')}")
            except Exception as e:
                self.logger(f"  ! Error processing {rgb_path.name}: {e}")
        if self.debug_out and master_list:
            master_json_path = self.debug_out / "_master_convergence_list.json"
            try:
                with open(master_json_path, "w", encoding="utf-8") as f:
                    json.dump(master_list, f, indent=4)
                self.logger(f"Saved master convergence list to {master_json_path}")
            except Exception as e:
                warnings.warn(f"Failed to write master convergence list: {e}")

    def _read_depth_frame(self, cap_depth, depth_path: Path, idx: int, size: Tuple[int, int]) -> Optional[np.ndarray]:
        if cap_depth:
            ret, frame = cap_depth.read()
            if not ret: return None
            depth_raw = frame
        else:
            img_files = sorted([p for p in depth_path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
            if idx >= len(img_files): return None
            depth_raw = cv2.imread(str(img_files[idx]), cv2.IMREAD_UNCHANGED)
        if depth_raw is None: return None
        if depth_raw.dtype == np.uint8: depth = depth_raw.astype(np.float32) / 255.0
        elif depth_raw.dtype == np.uint16: depth = depth_raw.astype(np.float32) / 65535.0
        else:
            depth = depth_raw.astype(np.float32)
            min_d, max_d = np.min(depth), np.max(depth)
            depth = (depth - min_d) / (max_d - min_d) if max_d > min_d else np.full_like(depth, 0.5)
        return cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)

    def _process_single(self, rgb_path: Path, depth_path: Path) -> float:
        cap_rgb = cv2.VideoCapture(str(rgb_path))
        if not cap_rgb.isOpened():
            warnings.warn(f"Failed to open RGB video {rgb_path}"); return 0.5
        cap_depth = None
        if depth_path.is_file():
            cap_depth = cv2.VideoCapture(str(depth_path))
            if not cap_depth.isOpened(): warnings.warn(f"Failed to open depth video {depth_path}"); cap_depth = None
        ret, frame = cap_rgb.read()
        if not ret: cap_rgb.release(); return 0.5
        height, width = frame.shape[:2]

        writer = self._init_debug_writer(rgb_path, width, height, cap_rgb.get(cv2.CAP_PROP_FPS))
        
        # --- Full CUDA Pipeline Setup ---
        use_cuda = self.device == "cuda" and cv2.cuda.getCudaEnabledDeviceCount() > 0
        flow_calculator, gpu_mats = None, {}
        if use_cuda:
            self.logger("Using full CUDA pipeline.")
            try:
                flow_calculator = cv2.cuda_FarnebackOpticalFlow.create(5, 0.5, False, 15, 3, 5, 1.2, 0)
                gpu_mats = {
                    'frame': cv2.cuda_GpuMat(), 'gray': cv2.cuda_GpuMat(),
                    'prev_gray': cv2.cuda_GpuMat(), 'flow_x': cv2.cuda_GpuMat(),
                    'flow_y': cv2.cuda_GpuMat(), 'mag': cv2.cuda_GpuMat(),
                }
            except Exception as e:
                warnings.warn(f"Failed to initialize CUDA components, falling back to CPU: {e}"); use_cuda = False
        # --- End Setup ---

        saliency = SaliencyModel((height, width))
        if self.tracker_name == "deepsort" and HAVE_DEEPSORT: self.tracker = DeepSort(max_age=30)
        else: self.tracker = SimpleTracker()
        
        track_depth_memory, depth_sequence, idx = {}, [], 0
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))

        while idx < frame_count:
            cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_rgb, frame_bgr = cap_rgb.read()
            if not ret_rgb: break
            depth_frame = self._read_depth_frame(cap_depth, depth_path, idx, (width, height))
            if depth_frame is None: break

            # --- Detection and Tracking (CPU) ---
            detections = self.detector.detect(frame_bgr) if self.detector else []
            if self.tracker_name == "deepsort" and HAVE_DEEPSORT and isinstance(self.tracker, DeepSort):
                ds_dets = [([x1, y1, x2 - x1, y2 - y1], conf, str(cls)) for x1, y1, x2, y2, conf, cls in detections]
                ds_tracks = self.tracker.update_tracks(ds_dets, frame=frame_bgr)
                current_ids = {int(t.track_id) for t in ds_tracks if t.is_confirmed()}
                tracks = [SimpleTrack(id=int(t.track_id), bbox=tuple(map(int, t.to_ltrb())), age=t.age, last_seen=t.time_since_update, prev_depth=track_depth_memory.get(int(t.track_id), 0.5)) for t in ds_tracks if t.is_confirmed()]
                for tid in list(track_depth_memory.keys()):
                    if tid not in current_ids: del track_depth_memory[tid]
            else:
                tracks = self.tracker.update(detections)

            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                roi = depth_frame[max(0, y1):min(height, y2)+1, max(0, x1):min(width, x2)+1]
                track.median_depth = float(np.median(roi)) if roi.size > 0 else 0.5
                if self.tracker_name == "deepsort": track_depth_memory[track.id] = track.median_depth
            
            # --- Scoring Pipeline ---
            raw_scores, final_scores, scene_context, flow, global_motion, heatmap = {}, [], "standard", None, 0.0, None
            if tracks:
                # --- Motion Calculation (CPU or GPU) ---
                if use_cuda:
                    gpu_mats['frame'].upload(frame_bgr)
                    cv2.cuda.cvtColor(gpu_mats['frame'], cv2.COLOR_BGR2GRAY, stream=cv2.cuda.Stream(), dst=gpu_mats['gray'])
                    if not gpu_mats['prev_gray'].empty():
                        gpu_flow = flow_calculator.calc(gpu_mats['prev_gray'], gpu_mats['gray'], None)
                        cv2.cuda.split(gpu_flow, (gpu_mats['flow_x'], gpu_mats['flow_y']))
                        cv2.cuda.magnitude(gpu_mats['flow_x'], gpu_mats['flow_y'], dst=gpu_mats['mag'])
                        mag = gpu_mats['mag'].download()
                        flow = gpu_flow.download()
                        global_motion = np.median(mag)
                    gpu_mats['gray'], gpu_mats['prev_gray'] = gpu_mats['prev_gray'], gpu_mats['gray'] # Swap
                else: # Fallback to CPU
                    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                    if 'prev_gray_cpu' in locals():
                        flow = cv2.calcOpticalFlowFarneback(locals()['prev_gray_cpu'], gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                        global_motion = np.median(mag)
                    locals()['prev_gray_cpu'] = gray
                # --- End Motion ---

                raw_scores['face'] = self.face_detector.score_faces(frame_bgr, tracks)
                raw_scores['composition'] = self._score_composition(tracks, (height, width))
                raw_scores['duration'] = [math.log(max(1, track.age)) for track in tracks]
                motion_scores = []
                if flow is not None:
                    for track in tracks:
                        x1, y1, x2, y2 = track.bbox
                        region_mag = mag[y1:y2+1, x1:x2+1]
                        local_motion = float(np.median(region_mag)) if region_mag.size > 0 else 0.0
                        rel_2d = max(0.0, local_motion - global_motion)
                        z_motion = max(0.0, track.prev_depth - track.median_depth)
                        motion_scores.append(rel_2d + z_motion * 50.0)
                else:
                    motion_scores = [0.0] * len(tracks)
                raw_scores['motion'] = motion_scores
                
                heatmap = saliency.compute_heatmap(frame_bgr)
                raw_scores['gaze'] = saliency.score(tracks, heatmap)
                has_large_face = any(s > 0.05 for s in raw_scores['face'])
                final_scores, scene_context = self._compute_final_scores(raw_scores, global_motion, has_large_face)

            # --- Depth Selection ---
            idx_max, primary_track = None, None
            if final_scores:
                idx_max = int(np.argmax(final_scores))
                primary_track = tracks[idx_max]
                median_depth = primary_track.median_depth
            else:
                median_depth = float(np.median(depth_frame))
            depth_sequence.append(median_depth)

            if writer:
                running_depth = float(np.median(self.smoother.smooth(depth_sequence))) if depth_sequence else None
                annotated = self._render_frame(frame_bgr, detections, tracks, idx_max, raw_scores, final_scores, heatmap, depth_frame, primary_track.bbox if primary_track else None, flow, global_motion, idx, median_depth, running_depth, scene_context)
                writer.write(annotated)
            
            idx += self.stride
        
        cap_rgb.release()
        if cap_depth: cap_depth.release()
        if writer: writer.release()
        
        if not depth_sequence:
            warnings.warn(f"No valid depth sequence for {rgb_path}; defaulting convergence to 0.5")
            return 0.5
        
        smoothed = self.smoother.smooth(depth_sequence)
        return float(max(0.0, min(1.0, np.median(smoothed)))) if smoothed else 0.5
    
    def _init_debug_writer(self, rgb_path: Path, w: int, h: int, fps_in: float) -> Optional[cv2.VideoWriter]:
        if not (self.debug_video and self.debug_out): return None
        out_w, out_h = max(1, int(w * self.debug_scale)), max(1, int(h * self.debug_scale))
        if not (fps_in and fps_in > 0): self.logger("[Debug] Invalid source FPS; debug video disabled."); return None
        
        debug_fps = float(fps_in) / self.stride
        base_name = f"{rgb_path.stem}__debug"
        attempts = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi")]
        for fourcc_name, ext in attempts:
            out_file = self.debug_out / f"{base_name}{ext}"
            writer = cv2.VideoWriter(str(out_file), cv2.VideoWriter_fourcc(*fourcc_name), debug_fps, (out_w, out_h))
            if writer.isOpened():
                self.logger(f"[Debug] Saving debug video to {out_file} ({fourcc_name}, {debug_fps:.2f}fps, {out_w}x{out_h})")
                return writer
        self.logger("[Debug] Could not open a debug video writer; debug video disabled.")
        return None

    def _render_frame(self, base, dets, tracks, p_idx, r_scores, f_scores, heatmap, depth, roi, flow, g_motion, f_idx, f_depth, r_depth, context) -> np.ndarray:
        canvas = base.copy()
        h, w = canvas.shape[:2]
        
        # Overlays
        if self.debug_layers is None or 'depth' in self.debug_layers:
            depth_vis = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
            cv2.addWeighted(cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET), 0.3, canvas, 0.7, 0, dst=canvas)
            if roi: cv2.rectangle(canvas, (roi[0], roi[1]), (roi[2], roi[3]), (255,255,255), 1, cv2.LINE_AA)
        if (self.debug_layers is None or 'saliency' in self.debug_layers) and heatmap is not None:
            sal_vis = (np.clip(heatmap, 0, 1) * 255).astype(np.uint8)
            cv2.addWeighted(cv2.applyColorMap(sal_vis, cv2.COLORMAP_JET), 0.3, canvas, 0.7, 0, dst=canvas)
        
        # Detections and Tracks
        if self.debug_layers is None or 'detections' in self.debug_layers:
            for x1, y1, x2, y2, conf, cid in dets: cv2.rectangle(canvas, (x1, y1), (x2, y2), (255,255,0), 1)
        if self.debug_layers is None or 'tracks' in self.debug_layers:
            for i, t in enumerate(tracks):
                color = ((t.id*37)%256, (t.id*17)%256, (t.id*29)%256)
                thick = 3 if i == p_idx else 2
                cv2.rectangle(canvas, (t.bbox[0], t.bbox[1]), (t.bbox[2], t.bbox[3]), color, thick)
                cv2.putText(canvas, f"ID{t.id}", (t.bbox[0], t.bbox[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # HUD
        hud_lines = [f"Frame: {f_idx}", f"Context: {context.upper()}", f"Depth: {f_depth:.3f}", f"Running Depth: {r_depth or 0:.3f}"]
        if p_idx is not None:
            hud_lines.extend(["-"*10, f"PRIMARY ID {tracks[p_idx].id}", f"SCORE: {f_scores[p_idx]:.2f}"])
        for i, line in enumerate(hud_lines):
            cv2.putText(canvas, line, (6, 16+i*14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (5, 15+i*14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        
        if w != base.shape[1] or h != base.shape[0]:
            canvas = cv2.resize(canvas, (base.shape[1], base.shape[0]), interpolation=cv2.INTER_AREA)
        return canvas

###########################################################################
# Command line interface
###########################################################################

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute narrative convergence values for videos.")
    parser.add_argument("--videos", type=str, required=True, help="Directory of RGB videos")
    parser.add_argument("--depths", type=str, required=True, help="Directory of depth videos/images")
    parser.add_argument("--tracker", type=str, default="deepsort", help="Tracker type")
    parser.add_argument("--detector", type=str, default="yolo12s", choices=["yolo12n", "yolo12s", "yolo12m"], help="Detector type")
    parser.add_argument("--gaze", type=str, default="auto", help="Gaze (saliency) model")
    parser.add_argument("--kalman", action="store_true", help="Use Kalman filter")
    parser.add_argument("--window", type=int, default=9, help="Moving average window size")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Computation device")
    parser.add_argument("--max_videos", type=int, help="Process at most N videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--debug-video", action="store_true", help="Enable export of annotated debug videos")
    parser.add_argument("--debug-out", type=str, help="Output directory for debug files")
    parser.add_argument("--debug-scale", type=float, default=1.0, help="Scale factor for debug video frames")
    parser.add_argument("--debug-layers", type=str, help="Comma-separated list of debug layers to render")
    args = parser.parse_args()

    video_dir, depth_dir = Path(args.videos), Path(args.depths)
    if not video_dir.is_dir() or not depth_dir.is_dir():
        print("Error: Video and depth directories must exist.", file=sys.stderr); sys.exit(1)
    
    debug_out_path = Path(args.debug_out) if args.debug_out else None
    if debug_out_path: ensure_dir(debug_out_path)
    if args.debug_video and not debug_out_path:
        print("Error: --debug-out is required when --debug-video is set.", file=sys.stderr); sys.exit(1)

    processor_args = {
        "detector_name": args.detector, "tracker_name": args.tracker, "gaze_name": args.gaze,
        "use_kalman": args.kalman, "window": args.window, "device": args.device,
        "max_videos": args.max_videos, "seed": args.seed, "verbose": args.verbose, "stride": args.stride,
        "debug_video": args.debug_video, "debug_out": debug_out_path, "debug_scale": args.debug_scale,
        "debug_layers": args.debug_layers.split(',') if args.debug_layers else None,
    }
    processor = NarrativeProcessor(video_dir=video_dir, depth_dir=depth_dir, **processor_args)
    processor.process()
    if args.verbose: print("Processing complete.")

if __name__ == "__main__":
    main()
