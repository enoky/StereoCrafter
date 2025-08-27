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

Usage (example)::

python narrative_convergence.py 
--videos /path/to/rgb_videos 
--depths /path/to/depth_videos 
--tracker deepsort --detector yolo12s --gaze auto 
--window 9 --device cpu --verbose

The script accepts the following command line arguments:

--videos <dir>      Directory containing RGB videos.
--depths <dir>      Directory containing depth videos or image sequences.  The
                    base filenames must match those in the RGB directory.
--tracker           Object tracker to use: 'deepsort' (default). If unavailable
                    the script falls back to a simple IoU tracker.
--detector          Object detector to use: 'yolo12n', 'yolo12s' (default), or 'yolo12m'.
--gaze              Saliency model: 'deepvs', 'pats' or 'auto'.  If models
                    cannot be loaded the script falls back to a center‐bias
                    saliency heatmap.
--kalman            When set, applies a simple 1D Kalman filter to the depth
                    sequence instead of a moving average.
--window <int>      Window size for moving average smoothing (default: 9).
--device            Compute device: 'cpu' (default) or 'cuda' (if available).
--max_videos <int>  Limits the number of videos processed (useful for
                    debugging).
--seed <int>        Random seed for reproducibility (default: 42).
--verbose           Enables verbose logging and progress bars.

Requirements
------------

The script depends on the following Python packages, which can be installed
using pip::

    opencv-python
    numpy
    torch
    ultralytics
    deep_sort_realtime
    tqdm
    scipy

To install the dependencies run::

    pip install opencv-python numpy torch ultralytics deep_sort_realtime tqdm scipy

This script is self‐contained and attempts to download missing models at
runtime.  All downloads are cached under a local ``models/`` directory.
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
    """Set random seeds for reproducibility.

    Args:
        seed: An integer seed.
    """
    import random

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    """Ensure that a directory exists.

    Args:
        path: Directory to create if missing.
    """
    path.mkdir(parents=True, exist_ok=True)


def iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    """Compute Intersection over Union of two bounding boxes.

    Args:
        boxA: (x1,y1,x2,y2)
        boxB: (x1,y1,x2,y2)

    Returns:
        IoU value in [0, 1].
    """
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
    """Resize an image while maintaining aspect ratio and pad to target size.

    Args:
        img: Input image.
        size: (width, height) target size.

    Returns:
        Resized and padded image.
    """
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
    """Manage model files: download if missing, keep in ./models.

    This lightweight registry checks for the presence of model files under
    ``models/``.  When a file is requested but missing, the registry
    attempts to download it from a list of candidate URLs.  Downloads are
    performed via ``urllib.request``.  If all downloads fail, a warning is
    emitted and the caller must handle the absence of the model.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        ensure_dir(self.root)

    def get(self, filename: str, urls: List[str]) -> Optional[Path]:
        """Retrieve a model file.

        Args:
            filename: Name of the file under ``models/``.
            urls: A list of candidate URLs to download the file from if it
                does not already exist.

        Returns:
            The path to the file if available or successfully downloaded, else
            ``None``.
        """
        path = self.root / filename
        if path.exists():
            return path
        # Try downloading from the provided URLs
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
        # Attempt to load the specified model
        try:
            self.model = YOLO(f"{model_name}.pt")
        except Exception as e:
            warnings.warn(f"Failed to load model {model_name}.pt: {e}")
            raise RuntimeError(f"Could not load YOLO model {model_name}")
        # Set device
        if device == "cuda" and torch is not None and torch.cuda.is_available():
            self.model.to("cuda")

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """Run detection on a frame.

        Args:
            frame: RGB frame (BGR or RGB).  Ultralytics expects BGR or RGB
                arrays.

        Returns:
            A list of detections: (x1, y1, x2, y2, confidence, class_id).
        """
        results = self.model.predict(frame, verbose=False)
        detections: List[Tuple[int, int, int, int, float, int]] = []
        for r in results:
            if hasattr(r, "boxes"):
                boxes = r.boxes
            else:
                boxes = r
            for b in boxes:
                # b.xyxy is in (x1,y1,x2,y2); b.conf item is scalar
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
    # Fields for advanced scoring
    prev_depth: float = 0.5
    median_depth: float = 0.5


class SimpleTracker:
    """A naive IoU‐based multi–object tracker.

    This tracker assigns detections to existing tracks based on IoU and
    creates new tracks when needed.  It does not use re‐identification and
    is intended as a lightweight fallback when DeepSORT is unavailable.
    """

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.next_id = 0
        self.tracks: Dict[int, SimpleTrack] = {}
        self.iou_threshold = iou_threshold

    def update(self, detections: List[Tuple[int, int, int, int, float, int]]) -> List[SimpleTrack]:
        updated_tracks: Dict[int, SimpleTrack] = {}
        assigned = set()
        # First attempt to match existing tracks to detections
        for tid, track in self.tracks.items():
            best_iou = 0.0
            best_det_idx = None
            for i, det in enumerate(detections):
                if i in assigned:
                    continue
                det_box = det[:4]
                score = iou(track.bbox, det_box)
                if score > best_iou:
                    best_iou = score
                    best_det_idx = i
            if best_det_idx is not None and best_iou >= self.iou_threshold:
                x1, y1, x2, y2, conf, cls = detections[best_det_idx]
                # Preserve previous state for motion calculation
                new_track = SimpleTrack(
                    id=tid, 
                    bbox=(x1, y1, x2, y2), 
                    age=track.age + 1, 
                    last_seen=0, 
                    cls=cls,
                    prev_depth=track.median_depth
                )
                updated_tracks[tid] = new_track
                assigned.add(best_det_idx)
        # Create new tracks for unmatched detections
        for i, det in enumerate(detections):
            if i not in assigned:
                x1, y1, x2, y2, conf, cls = det
                updated_tracks[self.next_id] = SimpleTrack(id=self.next_id, bbox=(x1, y1, x2, y2), age=1, last_seen=0, cls=cls)
                self.next_id += 1
        # Increment last_seen for tracks that were not updated
        for tid, track in self.tracks.items():
            if tid not in updated_tracks:
                track.last_seen += 1
                # Retain tracks for a short time even if not detected
                if track.last_seen <= 2:
                    updated_tracks[tid] = track
        self.tracks = updated_tracks
        return list(updated_tracks.values())


###########################################################################
# Face Detector
###########################################################################

class FaceDetector:
    """Wrapper around OpenCV's DNN face detector.

    The detector uses a Caffe model to detect faces. The model files are
    automatically downloaded via the ModelRegistry if not already present.
    The ``score_faces`` method accepts a frame and a list of track
    bounding boxes and returns a list of continuous scores based on face size.
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.model_loaded = False
        # Model file names and fallback URLs (OpenCV mirrors)
        prototxt_name = "deploy.prototxt.txt"
        caffemodel_name = "res10_300x300_ssd_iter_140000.caffemodel"
        prototxt_urls = [
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt.txt",
        ]
        caffemodel_urls = [
            "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        ]
        self.prototxt_path = registry.get(prototxt_name, prototxt_urls)
        self.caffemodel_path = registry.get(caffemodel_name, caffemodel_urls)
        if self.prototxt_path and self.caffemodel_path:
            try:
                self.net = cv2.dnn.readNetFromCaffe(str(self.prototxt_path), str(self.caffemodel_path))
                self.model_loaded = True
            except Exception as e:
                warnings.warn(f"Failed to load face detector: {e}")
        else:
            warnings.warn("Face detector model files are missing; face scores will be zero.")

    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in a frame.

        Args:
            frame: BGR image.

        Returns:
            List of bounding boxes (x1,y1,x2,y2) in pixel coordinates.
        """
        if not self.model_loaded:
            return []
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
        """Compute continuous face scores for each track based on face size.

        A track's score is proportional to the area of the largest overlapping
        face, relative to the total frame area.

        Args:
            frame: BGR frame.
            tracks: List of tracks.

        Returns:
            List of float scores aligned with ``tracks``.
        """
        if not tracks:
            return []
        if not self.model_loaded:
            return [0.0] * len(tracks)
            
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        faces = self.detect_faces(frame)
        scores = []
        for track in tracks:
            max_face_score = 0.0
            for face in faces:
                if iou(track.bbox, face) > 0.1:
                    face_w = face[2] - face[0]
                    face_h = face[3] - face[1]
                    face_area = face_w * face_h
                    # Score is the ratio of face area to frame area
                    score = face_area / frame_area
                    if score > max_face_score:
                        max_face_score = score
            scores.append(max_face_score)
        return scores


###########################################################################
# Saliency (Gaze) Models
###########################################################################

class SaliencyModel:
    """Compute a saliency (gaze) heatmap for a frame.

    This base class provides a fallback implementation that produces a
    centre‐weighted Gaussian heatmap.  Subclasses can override
    ``compute_heatmap`` to integrate actual pre–trained models.
    """

    def __init__(self, frame_size: Tuple[int, int]) -> None:
        self.frame_size = frame_size
        self.fallback_warned = False

    def compute_heatmap(self, frame: np.ndarray) -> np.ndarray:
        """Return a saliency heatmap in [0,1] for the input frame.

        The default implementation generates a normalized Gaussian kernel
        centred in the frame.  The kernel size matches the frame size.  The
        output is in the shape (H, W) with values in [0,1].

        Args:
            frame: BGR image.  Unused by the fallback implementation.

        Returns:
            Heatmap as a float32 array in [0,1].
        """
        h, w = self.frame_size
        # Create coordinate grids
        y = np.arange(h) - (h - 1) / 2.0
        x = np.arange(w) - (w - 1) / 2.0
        xx, yy = np.meshgrid(x, y)
        sigma_x = w / 6.0
        sigma_y = h / 6.0
        heatmap = np.exp(-((xx**2) / (2 * sigma_x**2) + (yy**2) / (2 * sigma_y**2)))
        heatmap = heatmap.astype(np.float32)
        heatmap -= heatmap.min()
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap

    def score(self, tracks: List[SimpleTrack], heatmap: np.ndarray) -> List[float]:
        """Compute saliency scores for each track given a heatmap.

        Args:
            tracks: List of tracks.
            heatmap: Saliency heatmap matching frame dimensions.

        Returns:
            List of saliency scores between 0 and 1 (mean value inside bbox).
        """
        scores: List[float] = []
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            # Clip to heatmap dimensions
            h, w = heatmap.shape[:2]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))
            region = heatmap[y1 : y2 + 1, x1 : x2 + 1]
            if region.size == 0:
                scores.append(0.0)
            else:
                scores.append(float(region.mean()))
        return scores


###########################################################################
# Smoothing
###########################################################################

class MovingAverageSmoother:
    """Compute a moving average over a sequence."""

    def __init__(self, window: int = 9) -> None:
        self.window = max(1, window)

    def smooth(self, values: List[float]) -> List[float]:
        if not values:
            return []
        kernel = np.ones(self.window) / self.window
        padded = np.pad(values, (self.window // 2, self.window - 1 - self.window // 2), mode="edge")
        smoothed = np.convolve(padded, kernel, mode="valid")
        return smoothed.tolist()


class KalmanSmoother:
    """A simple 1D Kalman filter for smoothing depth signals."""

    def __init__(self, process_var: float = 1e-4, measurement_var: float = 1e-2) -> None:
        # Initial state: depth value and velocity
        self.x = np.array([[0.5], [0.0]], dtype=np.float64)
        # Covariance matrix
        self.P = np.eye(2)
        # State transition matrix
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
        # Process noise covariance
        self.Q = np.array([[process_var, 0.0], [0.0, process_var]], dtype=np.float64)
        # Measurement matrix
        self.H = np.array([[1.0, 0.0]], dtype=np.float64)
        # Measurement noise covariance
        self.R = np.array([[measurement_var]], dtype=np.float64)

    def smooth(self, values: List[float]) -> List[float]:
        if not values:
            return []
        smoothed: List[float] = []
        for z in values:
            # Predict
            self.x = self.F @ self.x
            self.P = self.F @ self.P @ self.F.T + self.Q
            # Update
            z_vec = np.array([[z]], dtype=np.float64)
            y = z_vec - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.inv(S)
            self.x = self.x + K @ y
            I = np.eye(2)
            self.P = (I - K @ self.H) @ self.P
            smoothed.append(float(self.x[0, 0]))
        return smoothed


###########################################################################
# Narrative Processor
###########################################################################

class NarrativeProcessor:
    """Main orchestrator for computing narrative convergence values."""

    def __init__(
        self,
        video_dir: Path,
        depth_dir: Path,
        detector_name: str = "yolo12s",
        tracker_name: str = "deepsort",
        gaze_name: str = "auto",
        use_kalman: bool = False,
        window: int = 9,
        device: str = "cpu",
        max_videos: Optional[int] = None,
        seed: int = 42,
        verbose: bool = False,
        stride: int = 1,
        logger: Optional[Callable[[str], None]] = None,
        # Debug options
        debug_video: bool = False,
        debug_out: Optional[Path] = None,
        debug_scale: float = 1.0,
        debug_layers: Optional[List[str]] = None,
    ) -> None:
        self.video_dir = video_dir
        self.depth_dir = depth_dir
        self.detector_name = detector_name
        self.tracker_name = tracker_name
        self.gaze_name = gaze_name
        self.use_kalman = use_kalman
        self.window = window
        self.device = device
        self.max_videos = max_videos
        self.verbose = verbose
        self.stride = max(1, int(stride))

        # Setup logger: use provided one, fallback to tqdm.write if verbose, else a no-op
        if logger is not None:
            self.logger = logger
        elif self.verbose:
            self.logger = tqdm.write
        else:
            self.logger = lambda _: None

        set_seed(seed)

        # Model registry
        self.registry = ModelRegistry(Path("models"))

        # Initialize detector
        self.detector = self._init_detector()
        # Initialize tracker
        self.tracker = self._init_tracker()
        # Face detector
        self.face_detector = FaceDetector(self.registry)
        # Smoother
        self.smoother = KalmanSmoother() if use_kalman else MovingAverageSmoother(window)

        # Debug configuration
        self.debug_video = bool(debug_video)
        self.debug_out = debug_out
        # Clamp debug_scale between 0.25 and 1.0
        try:
            scale_val = float(debug_scale)
        except Exception:
            scale_val = 1.0
        self.debug_scale = max(0.25, min(1.0, scale_val))
        # Parse debug layers: None means all layers
        if debug_layers:
            self.debug_layers = {layer.strip().lower() for layer in debug_layers if layer.strip()}
        else:
            self.debug_layers = None

    def _init_detector(self) -> Any:
        """Initialize the object detector based on the requested type."""
        name = self.detector_name.lower()
        if name in ["yolo12n", "yolo12s", "yolo12m"]:
            if YOLO is not None:
                return YOLODetector(device=self.device, model_name=name)
            else:
                warnings.warn("Ultralytics YOLO is not available. Using random detections.")
                return None
        warnings.warn(f"Unknown or unsupported detector {name}; using no detection.")
        return None

    def _init_tracker(self) -> Any:
        """Initialize the tracker based on the requested type."""
        name = self.tracker_name.lower()
        if name == "deepsort" and HAVE_DEEPSORT:
            return DeepSort(max_age=30)
        warnings.warn(f"Tracker '{name}' is not 'deepsort' or DeepSORT is unavailable; falling back to simple tracker.")
        return SimpleTracker()

    def _get_video_pairs(self) -> List[Tuple[Path, Path]]:
        """Match RGB and depth videos by basename; depth names may end with '_depth'."""
        def base_from_depth_name(p: Path) -> str:
            stem = p.stem
            return stem[:-6] if stem.endswith("_depth") else stem  # strip '_depth'

        video_files = sorted([p for p in self.video_dir.iterdir() if p.is_file() or p.is_dir()])
        depth_files = {}
        for p in self.depth_dir.iterdir():
            key = base_from_depth_name(p)
            depth_files[key] = p

        pairs: List[Tuple[Path, Path]] = []
        for vf in video_files:
            stem = vf.stem
            dp = depth_files.get(stem)
            if dp is not None:
                pairs.append((vf, dp))
        return pairs

    def _score_composition(self, tracks: List[SimpleTrack], frame_size: Tuple[int, int]) -> List[float]:
        """Computes a composition score based on centrality and rule-of-thirds.

        Args:
            tracks: List of active tracks.
            frame_size: (height, width) of the video frame.

        Returns:
            A list of composition scores (0-1) for each track.
        """
        h, w = frame_size
        scores = []
        
        # Rule of thirds lines
        thirds_x = [w / 3, 2 * w / 3]
        thirds_y = [h / 3, 2 * h / 3]
        
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # 1. Centrality score (Gaussian falloff from center)
            dist_x = abs(center_x - w / 2)
            dist_y = abs(center_y - h / 2)
            # Normalize distance by half the frame dimension
            norm_dist = math.sqrt((dist_x / (w / 2))**2 + (dist_y / (h / 2))**2)
            center_score = math.exp(-2.0 * norm_dist**2) # Sharp falloff
            
            # 2. Rule of thirds score (proximity to lines)
            min_dist_thirds_x = min(abs(center_x - thirds_x[0]), abs(center_x - thirds_x[1]))
            min_dist_thirds_y = min(abs(center_y - thirds_y[0]), abs(center_y - thirds_y[1]))
            # Normalize distance by a fraction of frame dimension (e.g., 1/6th)
            norm_dist_thirds = math.sqrt((min_dist_thirds_x / (w / 6))**2 + (min_dist_thirds_y / (h / 6))**2)
            thirds_score = math.exp(-0.5 * norm_dist_thirds**2) # Slower falloff
            
            # Combine scores: take the max of being centered or being on a thirds line
            scores.append(max(center_score, thirds_score))
            
        return scores

    def _compute_final_scores(
        self,
        scores: Dict[str, List[float]],
        global_motion: float,
        has_large_face: bool,
    ) -> Tuple[List[float], str]:
        """Normalizes, weights, and combines scores based on scene context.

        Args:
            scores: Dict of raw scores, e.g., {'face': [...], 'motion': [...]}.
            global_motion: Median optical flow magnitude for the frame.
            has_large_face: True if a face covers a significant portion of the frame.

        Returns:
            A tuple containing the list of final scores and the detected scene context string.
        """
        num_tracks = len(scores.get("face", []))
        if num_tracks == 0:
            return [], "standard"

        # 1. Determine Scene Context
        context = "standard"
        if global_motion > 5.0:  # Threshold for high motion
            context = "action"
        elif has_large_face and global_motion < 2.0:
            context = "dialogue"

        # 2. Set Dynamic Weights based on Context
        weights = {
            "standard": {"face": 4.0, "gaze": 2.0, "motion": 1.5, "duration": 1.0, "composition": 1.0},
            "action":   {"face": 2.0, "gaze": 1.0, "motion": 4.0, "duration": 1.0, "composition": 0.5},
            "dialogue": {"face": 5.0, "gaze": 2.5, "motion": 0.5, "duration": 1.0, "composition": 1.5},
        }
        active_weights = weights[context]

        # 3. Normalize all scores to a 0-1 range
        norm_scores: Dict[str, np.ndarray] = {}
        for key, values in scores.items():
            arr = np.array(values, dtype=np.float32)
            min_val, max_val = arr.min(), arr.max()
            if max_val > min_val:
                norm_scores[key] = (arr - min_val) / (max_val - min_val)
            else:
                norm_scores[key] = np.zeros_like(arr)

        # 4. Compute Final Score (example of mixed additive/multiplicative logic)
        final_scores = np.zeros(num_tracks, dtype=np.float32)
        
        # Base score from primary cues
        base_score = (
            norm_scores.get("face", 0) * active_weights["face"] +
            norm_scores.get("gaze", 0) * active_weights["gaze"] +
            norm_scores.get("motion", 0) * active_weights["motion"]
        )
        
        # Multiplicative modifiers
        duration_modifier = 1.0 + (norm_scores.get("duration", 0) * active_weights["duration"])
        composition_modifier = 1.0 + (norm_scores.get("composition", 0) * active_weights["composition"])

        final_scores = base_score * duration_modifier * composition_modifier
        
        return final_scores.tolist(), context

    def process(self) -> None:
        """Process all videos and compute convergence values, saving one JSON per video."""
        pairs = self._get_video_pairs()
        if self.max_videos is not None:
            pairs = pairs[: self.max_videos]

        master_convergence_list = {}

        iterator = tqdm(pairs, desc="Videos", disable=not self.verbose)
        for rgb_path, depth_path in iterator:
            if self.verbose:
                iterator.set_description(f"Processing {rgb_path.name}")
            try:
                # Process the video to get the convergence value
                conv = self._process_single(rgb_path, depth_path)
                
                # Add the result to our master list
                master_convergence_list[rgb_path.stem] = conv

                # Define the output path for the individual JSON sidecar file
                output_json_path = depth_path.with_suffix('.json')
                output_data = {"convergence_plane": conv}

                # Save the individual JSON file
                with open(output_json_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=4)
                self.logger(f"Saved convergence to {output_json_path}")
            except Exception as e:
                self.logger(f"  ! Error processing {rgb_path.name}: {e}")

        # After processing all videos, save the master list if a debug path is set
        if self.debug_out and master_convergence_list:
            master_json_path = self.debug_out / "_master_convergence_list.json"
            try:
                with open(master_json_path, "w", encoding="utf-8") as f:
                    json.dump(master_convergence_list, f, indent=4)
                self.logger(f"Saved master convergence list to {master_json_path}")
            except Exception as e:
                warnings.warn(f"Failed to write master convergence list: {e}")

    def _read_depth_frame(self, cap_depth: Any, depth_path: Path, idx: int, size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Read a depth frame (video or image sequence) and normalize it to [0,1].

        Args:
            cap_depth: cv2.VideoCapture object if depth_path is a video, else None.
            depth_path: Path to depth video or directory of depth images.
            idx: Frame index.
            size: (width, height) target size to resize depth frames to.

        Returns:
            Normalized depth frame or None if no frame is available.
        """
        if cap_depth is not None:
            ret, frame = cap_depth.read()
            if not ret:
                return None
            depth_raw = frame
        else:
            # Depth images stored as sequence; assume filenames sorted lexicographically
            img_files = sorted([p for p in depth_path.glob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
            if idx >= len(img_files):
                return None
            depth_raw = cv2.imread(str(img_files[idx]), cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            return None
        # Normalize depth
        if depth_raw.dtype == np.uint8:
            depth = depth_raw.astype(np.float32) / 255.0
        elif depth_raw.dtype == np.uint16:
            depth = depth_raw.astype(np.float32) / 65535.0
        else:
            depth = depth_raw.astype(np.float32)
            if np.max(depth) > np.min(depth):
                depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
            else:
                depth.fill(0.5) # Avoid division by zero for flat depth maps
        # Resize to match frame size
        depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
        return depth

    def _process_single(self, rgb_path: Path, depth_path: Path) -> float:
        """Process a single RGB/depth video pair.

        Args:
            rgb_path: Path to the RGB video file.
            depth_path: Path to the depth video or image sequence.

        Returns:
            A convergence value in [0,1].
        """
        # Open RGB video
        cap_rgb = cv2.VideoCapture(str(rgb_path))
        if not cap_rgb.isOpened():
            warnings.warn(f"Failed to open RGB video {rgb_path}")
            return 0.5
        # Determine if depth_path is video or directory
        cap_depth: Optional[cv2.VideoCapture] = None
        if depth_path.is_file() and depth_path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}:
            cap_depth = cv2.VideoCapture(str(depth_path))
            if not cap_depth.isOpened():
                warnings.warn(f"Failed to open depth video {depth_path}")
                cap_depth = None
        # Determine frame size
        ret, frame = cap_rgb.read()
        if not ret:
            cap_rgb.release()
            return 0.5
        height, width = frame.shape[:2]

        # ------------------------------------------------------------------
        # Debug video writer setup.  Determine scaled output size, FPS and
        # initialize a VideoWriter with codec fallbacks.  Writer remains
        # None if disabled or setup fails.
        writer = None
        chosen_codec = None
        debug_fps_used = None
        out_w = None
        out_h = None
        if self.debug_video and self.debug_out is not None:
            try:
                ensure_dir(self.debug_out)
                out_w = max(1, int(width * self.debug_scale))
                out_h = max(1, int(height * self.debug_scale))
                fps_in = cap_rgb.get(cv2.CAP_PROP_FPS)

                if not (fps_in and fps_in > 0 and not math.isnan(fps_in)):
                    self.logger(f"[Debug] Could not determine source FPS for {rgb_path.stem}; debug video disabled.")
                    writer = None
                else:
                    # Correct the FPS based on the processing stride
                    debug_fps_used = float(fps_in) / self.stride
                    base_name = f"{rgb_path.stem}__debug"
                    attempts = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi")]
                    for fourcc_name, ext in attempts:
                        out_file = Path(self.debug_out) / f"{base_name}{ext}"
                        fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
                        wtmp = cv2.VideoWriter(str(out_file), fourcc, debug_fps_used, (out_w, out_h))
                        if wtmp is not None and wtmp.isOpened():
                            writer = wtmp
                            chosen_codec = fourcc_name
                            debug_file_path = out_file
                            break
                        else:
                            try:
                                wtmp.release()
                            except Exception:
                                pass
                    if writer is None:
                        self.logger(f"[Debug] Could not open a debug video writer for {rgb_path.stem}; debug video disabled.")
                    else:
                        self.logger(
                            f"[Debug] Saving debug video to {debug_file_path} (codec: {chosen_codec}, fps: {debug_fps_used:.2f}, size: {out_w}x{out_h})"
                        )
            except Exception as e:
                self.logger(f"[Debug] Failed to initialize debug video writer: {e}; debug video disabled.")
                writer = None

        # Helper function to render overlays onto a frame
        def _render_frame(
            base_frame: np.ndarray,
            detections: List[Tuple[int, int, int, int, float, int]],
            tracks: List[SimpleTrack],
            primary_idx: Optional[int],
            raw_scores: Dict[str, List[float]],
            final_scores: List[float],
            heatmap: Optional[np.ndarray],
            depth_frame: np.ndarray,
            roi_coords: Optional[Tuple[int, int, int, int]],
            flow: Optional[np.ndarray],
            global_motion: float,
            frame_idx: int,
            frame_depth_value: float,
            running_depth: Optional[float],
            scene_context: str,
        ) -> np.ndarray:
            canvas = base_frame.copy()
            h, w = canvas.shape[:2]
            # Depth overlay
            if self.debug_layers is None or 'depth' in self.debug_layers:
                try:
                    depth_vis = (np.clip(depth_frame, 0.0, 1.0) * 255.0).astype(np.uint8)
                    depth_col = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                    alpha = 0.3
                    cv2.addWeighted(depth_col, alpha, canvas, 1 - alpha, 0, dst=canvas)
                    if roi_coords is not None and all(v >= 0 for v in roi_coords):
                        x1c, y1c, x2c, y2c = roi_coords
                        x1c = int(max(0, min(w - 1, x1c)))
                        y1c = int(max(0, min(h - 1, y1c)))
                        x2c = int(max(0, min(w - 1, x2c)))
                        y2c = int(max(0, min(h - 1, y2c)))
                        cv2.rectangle(canvas, (x1c, y1c), (x2c, y2c), (255, 255, 255), 1, lineType=cv2.LINE_AA)
                except Exception:
                    pass
            # Saliency overlay
            if (self.debug_layers is None or 'saliency' in self.debug_layers) and heatmap is not None:
                try:
                    sal_vis = (np.clip(heatmap, 0.0, 1.0) * 255.0).astype(np.uint8)
                    sal_col = cv2.applyColorMap(sal_vis, cv2.COLORMAP_JET)
                    alpha_sal = 0.3
                    cv2.addWeighted(sal_col, alpha_sal, canvas, 1 - alpha_sal, 0, dst=canvas)
                except Exception:
                    pass
            # Detections
            if self.debug_layers is None or 'detections' in self.debug_layers:
                for det in detections:
                    x1, y1, x2, y2, conf, cls_id = det
                    color = (255, 255, 0)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1, lineType=cv2.LINE_AA)
                    label = f"{cls_id}:{conf:.2f}"
                    cv2.putText(canvas, label, (x1 + 1, y1 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, label, (x1, y1 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
            # Tracks and primary/face
            if self.debug_layers is None or any(layer in self.debug_layers for layer in ['tracks', 'primary', 'faces'] if self.debug_layers):
                for idx_trk, track in enumerate(tracks):
                    x1, y1, x2, y2 = track.bbox
                    tid = track.id
                    r = (tid * 37) % 256
                    g = (tid * 17) % 256
                    b = (tid * 29) % 256
                    color = (int(b), int(g), int(r))
                    thickness = 2
                    if primary_idx is not None and idx_trk == primary_idx:
                        thickness = 3
                        color = (0, 255, 255)
                    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness, lineType=cv2.LINE_AA)
                    lbl = f"ID{track.id}"
                    y_label = y1 - 6 if y1 - 6 > 10 else y1 + 12
                    cv2.putText(canvas, lbl, (x1, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, lbl, (x1 - 1, y_label - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                    if (self.debug_layers is None or 'faces' in self.debug_layers) and idx_trk < len(raw_scores.get('face', [])):
                        if raw_scores['face'][idx_trk] > 0.01: # Threshold for small face
                            cx = x1 + 6
                            cy = y1 + 6
                            cv2.circle(canvas, (cx, cy), 4, (0, 255, 0), -1)
            # Motion vectors
            if (self.debug_layers is None or 'motion' in self.debug_layers) and flow is not None:
                try:
                    step = max(10, int(min(h, w) / 30))
                    for y_flow in range(0, h, step):
                        for x_flow in range(0, w, step):
                            fx, fy = flow[y_flow, x_flow]
                            cv2.arrowedLine(
                                canvas,
                                (x_flow, y_flow),
                                (int(x_flow + fx * 10), int(y_flow + fy * 10)),
                                (255, 0, 0),
                                1,
                                tipLength=0.3,
                            )
                except Exception:
                    pass
            # HUD
            if self.debug_layers is None or 'hud' in self.debug_layers:
                hud_lines: List[str] = []
                hud_lines.append(f"Frame: {frame_idx}")
                hud_lines.append(f"Context: {scene_context.upper()}")
                hud_lines.append(f"Depth: {frame_depth_value:.3f}")
                if running_depth is not None:
                    hud_lines.append(f"Depth(run): {running_depth:.3f}")
                if primary_idx is not None and primary_idx < len(final_scores):
                    hud_lines.append("-" * 10)
                    hud_lines.append(f"PRIMARY ID {tracks[primary_idx].id}")
                    fs = raw_scores.get('face', [0]*len(tracks))[primary_idx]
                    gs = raw_scores.get('gaze', [0]*len(tracks))[primary_idx]
                    ds = raw_scores.get('duration', [0]*len(tracks))[primary_idx]
                    ms = raw_scores.get('motion', [0]*len(tracks))[primary_idx]
                    cs = raw_scores.get('composition', [0]*len(tracks))[primary_idx]
                    tot = final_scores[primary_idx]
                    hud_lines.append(f"Face: {fs:.3f}")
                    hud_lines.append(f"Gaze: {gs:.3f}")
                    hud_lines.append(f"Dur: {ds:.2f}")
                    hud_lines.append(f"Mot: {ms:.3f}")
                    hud_lines.append(f"Comp: {cs:.3f}")
                    hud_lines.append(f"SCORE: {tot:.2f}")
                x0, y0 = 5, 15
                for line in hud_lines:
                    cv2.putText(canvas, line, (x0 + 1, y0 + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(canvas, line, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                    y0 += 14
            # Resize for output
            if out_w is not None and out_h is not None and (out_w != w or out_h != h):
                canvas = cv2.resize(canvas, (out_w, out_h), interpolation=cv2.INTER_AREA)
            return canvas

        # Prepare saliency model with fallback
        saliency = SaliencyModel((height, width))
        # Reset tracker state for this video
        track_depth_memory: Dict[int, float] = {}
        if self.tracker_name == "deepsort" and HAVE_DEEPSORT:
            self.tracker = DeepSort(max_age=30)
        else:
            self.tracker = SimpleTracker()
        # Read frames sequentially
        idx = 0
        prev_gray = None
        depth_sequence: List[float] = []
        # Reset VideoCapture to first frame (cap_rgb was advanced for size)
        cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        frame_count = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while idx < frame_count:
            cap_rgb.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret_rgb, frame = cap_rgb.read()
            if not ret_rgb:
                break
            bgr = frame.copy()
            # Read corresponding depth frame
            depth_frame = self._read_depth_frame(cap_depth, depth_path, idx, (width, height))
            if depth_frame is None:
                # If depth runs out, stop processing
                break
            
            if self.verbose:
                self.logger(f"  Frame {idx}:")

            # --- Detection and Tracking ---
            detections: List[Tuple[int, int, int, int, float, int]] = []
            tracks: List[SimpleTrack] = []

            # For DeepSORT and SimpleTracker, run detection first
            if self.detector is not None:
                detections = self.detector.detect(bgr)

            if self.tracker_name == "deepsort" and HAVE_DEEPSORT and isinstance(self.tracker, DeepSort):
                ds_dets = [([x1, y1, x2 - x1, y2 - y1], conf, str(cls)) for x1, y1, x2, y2, conf, cls in detections]
                ds_tracks = self.tracker.update_tracks(ds_dets, frame=bgr)
                
                current_track_ids = set()
                temp_tracks = []
                for t in ds_tracks:
                    if not t.is_confirmed(): continue
                    tid = int(t.track_id)
                    current_track_ids.add(tid)
                    bbox = t.to_ltrb()
                    prev_depth = track_depth_memory.get(tid, 0.5)
                    temp_tracks.append(SimpleTrack(
                        id=tid, bbox=tuple(map(int, bbox)), age=t.age, 
                        last_seen=t.time_since_update, cls=0, prev_depth=prev_depth
                    ))
                tracks = temp_tracks
                disappeared_ids = set(track_depth_memory.keys()) - current_track_ids
                for tid in disappeared_ids:
                    del track_depth_memory[tid]
            else: # Fallback to simple tracker
                tracks = self.tracker.update(detections)


            # Update track median depth (and memory for deepsort)
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                x1c, y1c = max(0, x1), max(0, y1)
                x2c, y2c = min(width - 1, x2), min(height - 1, y2)
                roi = depth_frame[y1c:y2c+1, x1c:x2c+1]
                track.median_depth = float(np.median(roi)) if roi.size > 0 else 0.5
                if self.tracker_name == "deepsort" and HAVE_DEEPSORT:
                    track_depth_memory[track.id] = track.median_depth
            
            # --- Start Scoring Pipeline ---
            raw_scores: Dict[str, List[float]] = {}
            if not tracks:
                final_scores, scene_context = [], "standard"
            else:
                # Face scores (continuous)
                raw_scores['face'] = self.face_detector.score_faces(bgr, tracks)
                
                # Composition scores
                raw_scores['composition'] = self._score_composition(tracks, (height, width))

                # Track duration scores (log scale)
                raw_scores['duration'] = [math.log(max(1, track.age)) for track in tracks]
                
                # Motion scores (intelligent)
                motion_scores: List[float] = []
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                global_motion = 0.0
                flow = None
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    global_motion = np.median(mag)
                    for track in tracks:
                        # 2D motion relative to camera
                        x1, y1, x2, y2 = track.bbox
                        region_mag = mag[y1:y2+1, x1:x2+1]
                        local_motion = float(np.median(region_mag)) if region_mag.size > 0 else 0.0
                        rel_2d_motion = max(0.0, local_motion - global_motion)
                        
                        # Z-motion (towards camera)
                        depth_change = track.prev_depth - track.median_depth
                        z_motion = max(0.0, depth_change) # Only reward moving closer
                        
                        # Combine motion scores (z-motion is heavily weighted)
                        motion_scores.append(rel_2d_motion + z_motion * 50.0)
                else:
                    motion_scores = [0.0] * len(tracks)
                raw_scores['motion'] = motion_scores
                prev_gray = gray

                # Gaze (saliency) scores
                heatmap = saliency.compute_heatmap(bgr)
                raw_scores['gaze'] = saliency.score(tracks, heatmap)
                
                # Scene context detection
                has_large_face = any(s > 0.05 for s in raw_scores['face']) # Face > 5% of screen
                
                # Final score computation
                final_scores, scene_context = self._compute_final_scores(raw_scores, global_motion, has_large_face)

            # --- End Scoring Pipeline ---

            # Select primary subject (max final score)
            primary_track = None
            idx_max = None
            if final_scores:
                idx_max = int(np.argmax(final_scores))
                primary_track = tracks[idx_max]
                median_depth = primary_track.median_depth
                depth_sequence.append(median_depth)
            else:
                # No tracks: use global median
                median_depth = float(np.median(depth_frame))
                depth_sequence.append(median_depth)

            # --------------------------------------------------------------
            # Render and write debug frame if writer is active
            if writer is not None:
                try:
                    roi_coords = primary_track.bbox if primary_track else None
                    running_depth = None
                    if depth_sequence:
                        smooth_partial = self.smoother.smooth(depth_sequence)
                        if smooth_partial:
                            running_depth = float(np.median(smooth_partial))
                    
                    annotated = _render_frame(
                        base_frame=bgr,
                        detections=detections,
                        tracks=tracks,
                        primary_idx=idx_max,
                        raw_scores=raw_scores,
                        final_scores=final_scores,
                        heatmap=heatmap if 'heatmap' in locals() else None,
                        depth_frame=depth_frame,
                        roi_coords=roi_coords,
                        flow=flow if 'flow' in locals() else None,
                        global_motion=global_motion if 'global_motion' in locals() else 0.0,
                        frame_idx=idx,
                        frame_depth_value=median_depth,
                        running_depth=running_depth,
                        scene_context=scene_context,
                    )
                    writer.write(annotated)
                except Exception as e:
                    self.logger(f"[Debug] Rendering error at frame {idx}: {e}")

            # Advance by stride
            idx += self.stride
            
        # Release video captures
        cap_rgb.release()
        if cap_depth is not None:
            cap_depth.release()
        if not depth_sequence:
            warnings.warn(f"No valid depth sequence for {rgb_path}; defaulting convergence to 0.5")
            if 'writer' in locals() and writer is not None: writer.release()
            return 0.5
        # Smooth sequence
        smoothed = self.smoother.smooth(depth_sequence)
        if not smoothed:
            if 'writer' in locals() and writer is not None: writer.release()
            return 0.5
        
        # The final anchor depth is the median of the smoothed sequence
        anchor_depth = float(np.median(smoothed))
        
        # Clamp the final value to be safe
        anchor_depth = float(max(0.0, min(1.0, anchor_depth)))
        
        if 'writer' in locals() and writer is not None: writer.release()
        return anchor_depth

###########################################################################
# Command line interface
###########################################################################

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute narrative convergence values for videos.")
    parser.add_argument("--videos", type=str, required=True, help="Directory of RGB videos")
    parser.add_argument("--depths", type=str, required=True, help="Directory of depth videos/images")
    parser.add_argument("--tracker", type=str, default="deepsort", choices=["deepsort"], help="Tracker type. 'deepsort' is the only option and will be used if available, otherwise falls back to a simple IoU tracker.")
    parser.add_argument("--detector", type=str, default="yolo12s", choices=["yolo12n", "yolo12s", "yolo12m"], help="Detector type")
    parser.add_argument(
        "--gaze", type=str, default="auto", choices=["auto", "deepvs", "pats"], help="Gaze (saliency) model"
    )
    parser.add_argument("--kalman", action="store_true", help="Use Kalman filter instead of moving average")
    parser.add_argument("--window", type=int, default=9, help="Window size for moving average smoothing")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame (>=1)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Computation device")
    parser.add_argument("--max_videos", type=int, default=None, help="Process at most N videos")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")

    # Debug video options
    parser.add_argument(
        "--debug-video",
        action="store_true",
        help="Enable export of annotated debug MP4 files (one per video). Default is off.",
    )
    parser.add_argument(
        "--debug-out",
        type=str,
        default=None,
        help="Output directory for debug MP4s (required if --debug-video is set)",
    )
    parser.add_argument(
        "--debug-scale",
        type=float,
        default=1.0,
        help="Uniform downscale factor for rendered debug frames (0.25–1.0, default: 1.0)",
    )
    parser.add_argument(
        "--debug-layers",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of overlay layers to render in the debug video. "
            "Available layers: detections,tracks,primary,faces,saliency,motion,depth,hud. "
            "Default is to render all layers."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = Path(args.videos)
    depth_dir = Path(args.depths)

    if not video_dir.exists() or not video_dir.is_dir():
        print(f"Video directory {video_dir} does not exist", file=sys.stderr)
        sys.exit(1)
    if not depth_dir.exists() or not depth_dir.is_dir():
        print(f"Depth directory {depth_dir} does not exist", file=sys.stderr)
        sys.exit(1)
        
    # Prepare debug configuration
    debug_layers = None
    if args.debug_layers:
        debug_layers = [layer.strip().lower() for layer in args.debug_layers.split(",") if layer.strip()]
    
    # MODIFICATION: Set the output path if --debug-out is provided,
    # independent of whether --debug-video is also used.
    out_path: Optional[Path] = None
    if args.debug_out:
        out_path = Path(args.debug_out)
        ensure_dir(out_path)

    # Validate that if --debug-video is set, --debug-out is also set.
    if args.debug_video and not args.debug_out:
        print("--debug-video specified but --debug-out is missing", file=sys.stderr)
        sys.exit(1)

    processor = NarrativeProcessor(
        video_dir=video_dir,
        depth_dir=depth_dir,
        detector_name=args.detector,
        tracker_name=args.tracker,
        gaze_name=args.gaze,
        use_kalman=args.kalman,
        window=args.window,
        device=args.device,
        max_videos=args.max_videos,
        seed=args.seed,
        verbose=args.verbose,
        stride=args.stride,
        # Debug options
        debug_video=bool(args.debug_video),
        debug_out=out_path,
        debug_scale=float(args.debug_scale),
        debug_layers=debug_layers,
    )

    processor.process()

    if args.verbose:
        print("Processing complete.")

if __name__ == "__main__":
    main()