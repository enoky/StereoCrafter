import os
import json
import logging
import tkinter as tk
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger(__name__)

class SidecarConfigManager:
    """Handles reading, writing, and merging of stereocrafter sidecar files."""

    # CENTRAL KEY MAP: {JSON_KEY: (Python_Type, Default_Value)}
    SIDECAR_KEY_MAP = {
        "convergence_plane": (float, 0.5),
        "max_disparity": (float, 20.0),
        "true_max": (float, None),
        "dp_total_max_true": (float, None),
        "dp_total_max_est": (float, None),
        "gamma": (float, 1.0),
        "input_bias": (float, 0.0),
        "depth_dilate_size_x": (float, 0.0),
        "depth_dilate_size_y": (float, 0.0),
        "depth_blur_size_x": (float, 0.0),
        "depth_blur_size_y": (float, 0.0),
        "depth_dilate_left": (float, 0.0),
        "depth_blur_left": (float, 0.0),
        "depth_blur_left_mix": (float, 0.5),
        "selected_depth_map": (str, ""),
        "left_border": (float, 0.0),
        "right_border": (float, 0.0),
        "border_mode": (str, None),
        "auto_border_L": (float, None),
        "auto_border_R": (float, None),
    }

    def _get_defaults(self) -> dict:
        """Returns a dictionary populated with all default values."""
        defaults = {}
        for key, (_, default_val) in self.SIDECAR_KEY_MAP.items():
            defaults[key] = default_val
        return defaults

    def load_sidecar_data(self, file_path: str) -> dict:
        """Loads and validates sidecar data, returning a dictionary merged with defaults."""
        data = self._get_defaults()
        if not os.path.exists(file_path):
            return data

        try:
            with open(file_path, "r") as f:
                sidecar_json = json.load(f)

            for key, (expected_type, _) in self.SIDECAR_KEY_MAP.items():
                if key in sidecar_json:
                    val = sidecar_json[key]
                    try:
                        if expected_type is int:
                            data[key] = int(val)
                        elif expected_type is float:
                            data[key] = float(val)
                        else:
                            data[key] = val
                    except (ValueError, TypeError):
                        pass

            # Preserve unknown keys for legacy migration
            for key, val in sidecar_json.items():
                if key not in data:
                    data[key] = val

        except Exception as e:
            logger.error(f"Failed to read/parse sidecar at {file_path}: {e}")

        return data

    def save_sidecar_data(self, file_path: str, data: dict) -> bool:
        """Saves a dictionary to the sidecar file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            output_data = {}
            for key, (expected_type, _) in self.SIDECAR_KEY_MAP.items():
                if key in data:
                    v = data[key]
                    if v is None:
                        continue
                    output_data[key] = v

            # Include any extra keys not in the map
            for key, val in data.items():
                if key not in output_data and val is not None:
                    output_data[key] = val

            with open(file_path, "w") as f:
                json.dump(output_data, f, indent=4)
            return True
        except Exception as e:
            logger.error(f"Failed to save sidecar to {file_path}: {e}")
            return False

    def get_merged_config(self, sidecar_path: str, gui_config: dict, override_keys: List[str]) -> dict:
        """Merges sidecar data with GUI configuration, allowing specific keys to be overridden."""
        merged_config = self.load_sidecar_data(sidecar_path)
        for key in override_keys:
            if key in gui_config and key in self.SIDECAR_KEY_MAP:
                expected_type = self.SIDECAR_KEY_MAP[key][0]
                try:
                    val = gui_config[key]
                    if expected_type is float:
                        merged_config[key] = float(val)
                    elif expected_type is int:
                        merged_config[key] = int(val)
                    else:
                        merged_config[key] = val
                except (ValueError, TypeError):
                    pass
        return merged_config

    # --- NEW: GUI Integration Methods ---

    def resolve_sidecar_path(self, depth_map_path: str, sidecar_folder: str, extension: str = ".fssidecar") -> str:
        """Determines the full path for the sidecar file."""
        if not depth_map_path:
            return ""
        depth_map_basename = os.path.splitext(os.path.basename(depth_map_path))[0]
        return os.path.join(sidecar_folder, f"{depth_map_basename}{extension}")

    def sync_to_gui(self, sidecar_data: Dict[str, Any], tk_vars: Dict[str, Any], mapping: Optional[Dict[str, str]] = None) -> None:
        """Updates tkinter variables from sidecar data using the key map or a custom mapping."""
        for key, val in sidecar_data.items():
            attr_name = mapping.get(key, f"{key}_var") if mapping else f"{key}_var"
            if attr_name in tk_vars:
                var = tk_vars[attr_name]
                try:
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(val))
                    elif isinstance(var, (tk.IntVar, tk.DoubleVar)):
                        var.set(float(val))
                    else:
                        var.set(str(val))
                except Exception:
                    pass

    def capture_from_gui(self, tk_vars: Dict[str, Any], mapping: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Captures values from tkinter variables into a data dictionary using a custom mapping."""
        data = {}
        for key in self.SIDECAR_KEY_MAP:
            attr_name = mapping.get(key, f"{key}_var") if mapping else f"{key}_var"
            if attr_name in tk_vars:
                var = tk_vars[attr_name]
                try:
                    data[key] = var.get()
                except Exception:
                    pass
        return data

    def calculate_borders_from_width_bias(self, width: float, bias: float) -> Tuple[float, float]:
        """Convert GUI Width/Bias to Left/Right border values."""
        if bias <= 0:
            left_b = width
            right_b = width * (1.0 + bias)
        else:
            right_b = width
            left_b = width * (1.0 - bias)
        return left_b, right_b

    def calculate_width_bias_from_borders(self, left_b: float, right_b: float) -> Tuple[float, float]:
        """Convert Left/Right border values back to GUI Width/Bias format."""
        w = max(left_b, right_b)
        if w > 0:
            if left_b > right_b:
                b = (right_b / left_b) - 1.0
            elif right_b > left_b:
                b = 1.0 - (left_b / right_b)
            else:
                b = 0.0
        else:
            b = 0.0
        return w, b

def find_sidecar_file(base_path: str) -> Optional[str]:
    """Looks for a sidecar JSON file next to the video file."""
    sidecar_path = f"{os.path.splitext(base_path)[0]}.fssidecar"
    if os.path.exists(sidecar_path):
        return sidecar_path
    json_path = f"{os.path.splitext(base_path)[0]}.json"
    if os.path.exists(json_path):
        return json_path
    return None

def find_sidecar_in_folder(folder: str, core_name: str) -> Optional[str]:
    """Looks for a sidecar file in a specific folder."""
    fssidecar_path = os.path.join(folder, f"{core_name}.fssidecar")
    if os.path.exists(fssidecar_path):
        return fssidecar_path
    json_path = os.path.join(folder, f"{core_name}.json")
    if os.path.exists(json_path):
        return json_path
    return None

def read_clip_sidecar(
    sidecar_manager: SidecarConfigManager,
    video_path: str,
    core_name: str,
    search_folders: Optional[List[str]] = None,
) -> dict:
    """Reads the sidecar file for a clip, checking multiple locations."""
    if search_folders:
        for folder in search_folders:
            sidecar_path = find_sidecar_in_folder(folder, core_name)
            if sidecar_path:
                return sidecar_manager.load_sidecar_data(sidecar_path)

    sidecar_path = find_sidecar_file(video_path)
    if sidecar_path:
        return sidecar_manager.load_sidecar_data(sidecar_path)

    return sidecar_manager._get_defaults()

def find_video_by_core_name(folder: str, core_name: str) -> Optional[str]:
    """Scans a folder for a file matching the core_name with any common video extension."""
    video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm")
    for ext in video_extensions:
        full_path = os.path.join(folder, f"{core_name}{ext[1:]}")
        if os.path.exists(full_path):
            return full_path
    return None
