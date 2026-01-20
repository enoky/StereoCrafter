import os
import json
import shutil
import threading
import tkinter as tk  # Required for Tooltip class
from tkinter import Toplevel, Label, ttk
from typing import Optional, Tuple, Callable
import logging

import numpy as np
import torch
from decord import VideoReader, cpu
import subprocess
import cv2
import gc
import time

VERSION = "26-01-20.3"

# --- Configure Logging ---
# Only configure basic logging if no handlers are already set up.
# This prevents duplicate log messages if a calling script configures logging independently.
if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S"
    )
    # logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# --- Slider default tick marker appearance (tweak if desired) ---

DEFAULT_TICK_RELY = 0.6  # moves tick up and down
DEFAULT_TICK_RELHEIGHT = 0.6  # tick length
DEFAULT_TICK_WIDTH = 2  # tick thickness
DEFAULT_TICK_COLOR = "#6b7280"  # "#ff0000" → red, "#00ff00" → green, "#0000ff" → blue,
# "#6b7280" → cool slate gray, "#5f6368" → G**gle-style dark gray, "#4b5563" → darker slate

# Horizontal alignment tweaks for default tick markers:
# - DEFAULT_TICK_TRACK_PAD_PCT nudges ticks inward from both ends of the trough (0.0 = no pad).
# - DEFAULT_TICK_X_OFFSET_PX applies a final pixel offset (positive = right, negative = left).
DEFAULT_TICK_TRACK_PAD_PCT = 0.0
DEFAULT_TICK_X_OFFSET_PX = 5

# --- Global Flags ---
CUDA_AVAILABLE = False


class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.show_delay = 600  # milliseconds
        self.hide_delay = 100  # milliseconds
        self.enter_id = None
        self.leave_id = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<ButtonPress>", self.hide_tooltip)  # Hide on click

    def _display_tooltip(self):
        if self.tooltip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 20
        self.tooltip_window = Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = Label(
            self.tooltip_window,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            justify="left",
            wraplength=250,
        )
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.enter_id:
            self.widget.after_cancel(self.enter_id)
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

    def show_tooltip(self, event=None):
        if self.leave_id:
            self.widget.after_cancel(self.leave_id)
        self.enter_id = self.widget.after(self.show_delay, self._display_tooltip)


class SidecarConfigManager:
    """Handles reading, writing, and merging of stereocrafter sidecar files."""

    # 1. CENTRAL KEY MAP: {JSON_KEY: (Python_Type, Default_Value)}
    # NOTE: Decimal places removed, as rounding is now handled by the GUI slider
    SIDECAR_KEY_MAP = {
        "convergence_plane": (float, 0.5),
        "max_disparity": (float, 20.0),
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
        # Add future keys here
    }

    def _get_defaults(self) -> dict:
        """Returns a dictionary populated with all default values."""
        defaults = {}
        # Iterate over the new map structure: key, (expected_type, default_val)
        for key, (_, default_val) in self.SIDECAR_KEY_MAP.items():
            defaults[key] = default_val
        return defaults

    def get_merged_config(
        self, sidecar_path: str, gui_config: dict, override_keys: list
    ) -> dict:
        """
        Merges sidecar data with GUI configuration, allowing specific keys to
        be overridden by GUI values.

        gui_config must use the same JSON keys as the sidecar file.
        """
        # 1. Load the sidecar data (base config, merged with defaults)
        merged_config = self.load_sidecar_data(sidecar_path)

        # 2. Apply GUI overrides
        for key in override_keys:
            if key in gui_config and key in self.SIDECAR_KEY_MAP:
                # Get the expected type from the map
                expected_type = self.SIDECAR_KEY_MAP[key][0]

                # Attempt to cast the GUI value to the expected type
                try:
                    val = gui_config[key]
                    if expected_type is float:
                        merged_config[key] = float(val)
                    elif expected_type is int:
                        merged_config[key] = int(val)
                    else:
                        merged_config[key] = val
                except (ValueError, TypeError):
                    logger.warning(
                        f"GUI value for '{key}' is invalid ({gui_config[key]}). Skipping override."
                    )

        return merged_config

    def load_sidecar_data(self, file_path: str) -> dict:
        """
        Loads and validates sidecar data, returning a dictionary merged with defaults.
        Returns defaults if file is not found or invalid.
        """
        data = self._get_defaults()
        if not os.path.exists(file_path):
            logger.debug(f"Sidecar not found at {file_path}. Returning defaults.")
            return data

        try:
            with open(file_path, "r") as f:
                sidecar_json = json.load(f)

            # Iterate over the new map structure: key, (expected_type, default_val)
            for key, (expected_type, _) in self.SIDECAR_KEY_MAP.items():
                if key in sidecar_json:
                    val = sidecar_json[key]
                    try:
                        # Attempt to cast the value to the expected type
                        if expected_type is int:
                            data[key] = int(val)
                        elif expected_type is float:
                            data[key] = float(val)
                        else:
                            data[key] = val
                    except (ValueError, TypeError):
                        logger.warning(
                            f"Sidecar key '{key}' has invalid value/type. Using default."
                        )

            # Preserve unknown keys for legacy migration (e.g., manual_border)
            for key, val in sidecar_json.items():
                if key not in data:
                    data[key] = val

        except Exception as e:
            logger.error(f"Failed to read/parse sidecar at {file_path}: {e}")
            # Still return defaults + whatever valid data was read before the failure

        return data

    def save_sidecar_data(self, file_path: str, data: dict) -> bool:
        """
        Saves a dictionary to the sidecar file, ensuring the directory and file are created.
        No rounding is applied here, assuming input data is pre-rounded.
        """
        try:
            # 1. Ensure directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # 2. Filter data (No rounding step needed)
            output_data = {}
            # Iterate over the new map structure: key, (expected_type, default_val)
            for key, (expected_type, _) in self.SIDECAR_KEY_MAP.items():
                if key in data:
                    output_data[key] = data[key]

            # 3. Write to file (mode 'w' creates the file if it doesn't exist)
            with open(file_path, "w") as f:
                json.dump(output_data, f, indent=4)

            logger.debug(f"Sidecar saved successfully to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save sidecar to {file_path}: {e}")
            return False


def apply_color_transfer(
    source_frame: torch.Tensor, target_frame: torch.Tensor
) -> torch.Tensor:
    """
    Transfers the color statistics from the source_frame to the target_frame using LAB color space.
    Expects source_frame and target_frame in [C, H, W] float [0, 1] format on CPU.
    Returns the color-adjusted target_frame in [C, H, W] float [0, 1] format.
    """
    try:
        # Ensure tensors are on CPU and convert to numpy arrays in HWC format
        # --- FIX: Squeeze the batch dimension if it exists ---
        source_for_permute = (
            source_frame.squeeze(0) if source_frame.dim() == 4 else source_frame
        )
        target_for_permute = (
            target_frame.squeeze(0) if target_frame.dim() == 4 else target_frame
        )

        source_np = source_for_permute.permute(1, 2, 0).numpy()  # [H, W, C]
        target_np = target_for_permute.permute(1, 2, 0).numpy()  # [H, W, C]
        # --- END FIX ---

        # Scale from [0, 1] to [0, 255] and convert to uint8
        source_np_uint8 = (np.clip(source_np, 0.0, 1.0) * 255).astype(np.uint8)
        target_np_uint8 = (np.clip(target_np, 0.0, 1.0) * 255).astype(np.uint8)

        # Convert to LAB color space
        source_lab = cv2.cvtColor(source_np_uint8, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target_np_uint8, cv2.COLOR_RGB2LAB)

        src_mean, src_std = cv2.meanStdDev(source_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

        src_mean = src_mean.flatten()
        src_std = src_std.flatten()
        tgt_mean = tgt_mean.flatten()
        tgt_std = tgt_std.flatten()

        # Ensure no division by zero
        src_std = np.clip(src_std, 1e-6, None)
        tgt_std = np.clip(tgt_std, 1e-6, None)

        target_lab_float = target_lab.astype(np.float32)
        for i in range(3):  # For L, A, B channels
            target_lab_float[:, :, i] = (
                target_lab_float[:, :, i] - tgt_mean[i]
            ) / tgt_std[i] * src_std[i] + src_mean[i]

        adjusted_lab_uint8 = np.clip(target_lab_float, 0, 255).astype(np.uint8)
        adjusted_rgb = cv2.cvtColor(adjusted_lab_uint8, cv2.COLOR_LAB2RGB)
        return torch.from_numpy(adjusted_rgb).permute(2, 0, 1).float() / 255.0
    except Exception as e:
        logger.error(
            f"Error during color transfer: {e}. Returning original target frame.",
            exc_info=True,
        )
        return target_frame


def apply_dubois_anaglyph(
    left_rgb_np: np.ndarray, right_rgb_np: np.ndarray
) -> np.ndarray:
    """
    Apply Dubois least-squares anaglyph transformation.
    Expects input as HWC NumPy arrays (uint8, 0-255).
    Returns HWC NumPy array (uint8, 0-255).
    """
    left_float = left_rgb_np.astype(np.float32) / 255.0
    right_float = right_rgb_np.astype(np.float32) / 255.0

    # Dubois red-cyan matrices (from splatting_gui)
    left_matrix = np.array(
        [
            [0.456, 0.500, 0.176],  # Left contributes to Red
            [-0.040, -0.038, -0.016],  # Left minimal to Green
            [-0.015, -0.021, -0.005],  # Left minimal to Blue
        ],
        dtype=np.float32,
    )

    right_matrix = np.array(
        [
            [-0.043, -0.088, -0.002],  # Right minimal to Red
            [0.378, 0.734, -0.018],  # Right contributes to Green
            [-0.072, -0.113, 1.226],  # Right contributes to Blue
        ],
        dtype=np.float32,
    )

    H, W = left_float.shape[:2]
    left_flat = left_float.reshape(-1, 3)
    right_flat = right_float.reshape(-1, 3)

    left_transformed = np.dot(left_flat, left_matrix.T)
    right_transformed = np.dot(right_flat, right_matrix.T)

    anaglyph_flat = np.clip(left_transformed + right_transformed, 0.0, 1.0)
    anaglyph_rgb = anaglyph_flat.reshape(H, W, 3)

    return (anaglyph_rgb * 255.0).astype(np.uint8)


def apply_optimized_anaglyph(
    left_rgb_np: np.ndarray, right_rgb_np: np.ndarray
) -> np.ndarray:
    """
    Apply Optimized Half-Color (minimal ghosting) anaglyph transformation.
    Expects input as HWC NumPy arrays (uint8, 0-255).
    Returns HWC NumPy array (uint8, 0-255).
    """
    left_float = left_rgb_np.astype(np.float32) / 255.0
    right_float = right_rgb_np.astype(np.float32) / 255.0

    # Optimized matrices for minimal eye strain
    left_matrix = np.array(
        [
            [0.0, 0.7, 0.3],  # Left contributes to Red (G+B weighted)
            [0.0, 0.0, 0.0],  # No green
            [0.0, 0.0, 0.0],  # No blue
        ],
        dtype=np.float32,
    )

    right_matrix = np.array(
        [
            [0.0, 0.0, 0.0],  # No red
            [0.0, 1.0, 0.0],  # Full green from right
            [0.0, 0.0, 1.0],  # Full blue from right
        ],
        dtype=np.float32,
    )

    H, W = left_float.shape[:2]
    left_flat = left_float.reshape(-1, 3)
    right_flat = right_float.reshape(-1, 3)

    left_transformed = np.dot(left_flat, left_matrix.T)
    right_transformed = np.dot(right_flat, right_matrix.T)

    anaglyph_flat = np.clip(left_transformed + right_transformed, 0.0, 1.0)
    anaglyph_rgb = anaglyph_flat.reshape(H, W, 3)

    return (anaglyph_rgb * 255.0).astype(np.uint8)


def create_single_slider_with_label_updater(
    GUI_self,
    parent: ttk.Frame,
    text: str,
    var: tk.Variable,
    from_: float,
    to: float,
    row: int,
    decimals: int = 0,
    tooltip_key: Optional[str] = None,
    trough_increment: float = -1.0,
    display_next_odd_integer: bool = False,
    custom_label_formula: Optional[Callable] = None,
    step_size: Optional[float] = None,
    default_value: Optional[float] = None,
) -> None:
    """
    Creates a single slider using Discrete Step Mapping.
    FIXED: Uses actual_step for all internal math to prevent disappearing labels
    when step_size is not explicitly provided (Blur/Dilation).
    """
    VALUE_LABEL_FIXED_WIDTH = 5

    label = ttk.Label(parent, text=text, anchor="e")
    label.grid(row=row, column=0, sticky="ew", padx=0, pady=2)

    if tooltip_key and hasattr(GUI_self, "_create_hover_tooltip"):
        GUI_self._create_hover_tooltip(label, tooltip_key)

    # REVERT/FIX: Use actual_step for all calculations.
    # If no step_size is passed, it correctly defaults to your original 0.5/1.0 logic.
    actual_step = step_size if step_size is not None else (0.5 if decimals > 0 else 1.0)

    total_steps = int((to - from_) / actual_step)
    internal_int_var = tk.IntVar(value=int((float(var.get()) - from_) / actual_step))

    def update_label_only(value_float: float) -> None:
        try:
            if custom_label_formula:
                value_label.config(text=custom_label_formula(value_float))
                return

            display_value = value_float
            if display_next_odd_integer:
                k_int = int(round(value_float))
                if k_int > 0 and k_int % 2 == 0:
                    display_value = k_int + 1
                elif k_int > 0:
                    display_value = k_int
                elif k_int == 0:
                    display_value = 0

            value_label.config(text=f"{display_value:.{decimals}f}")
        except Exception:
            pass

    def on_slider_move(val):
        notch = int(float(val))
        # Use actual_step to ensure math works for all sliders
        actual_val = from_ + (notch * actual_step)
        actual_val = max(from_, min(to, actual_val))

        var.set(actual_val)
        update_label_only(actual_val)

    slider = ttk.Scale(
        parent,
        from_=0,
        to=total_steps,
        variable=internal_int_var,
        orient="horizontal",
        command=on_slider_move,
    )
    slider.grid(row=row, column=1, sticky="ew", padx=2)

    value_label = ttk.Label(parent, text="", width=VALUE_LABEL_FIXED_WIDTH)
    value_label.grid(row=row, column=2, sticky="w", padx=0)
    parent.grid_columnconfigure(1, weight=1)

    slider.bind("<ButtonRelease-1>", GUI_self.on_slider_release)

    def sync_external_change():
        try:
            current_f = float(var.get())
            # Use actual_step for synchronization
            new_notch = int((current_f - from_) / actual_step)
            internal_int_var.set(new_notch)
            update_label_only(current_f)
        except Exception:
            pass

    sync_external_change()

    # --- Default marker & right-click reset (does not affect layout) ---
    if default_value is not None:

        def _reset_to_default(event=None):
            try:
                var.set(default_value)
                sync_external_change()

                # Keep UX consistent with a normal slider release: refresh preview immediately.
                try:
                    if hasattr(GUI_self, "on_slider_release"):
                        GUI_self.on_slider_release(None)
                    elif hasattr(GUI_self, "update_preview_from_controls"):
                        GUI_self.update_preview_from_controls()
                except Exception:
                    pass
            except Exception:
                pass
            return "break"

        # Right-click reset (Windows: Button-3; some systems: Button-2)
        for _w in (slider, value_label):
            try:
                _w.bind("<Button-3>", _reset_to_default)
                _w.bind("<Button-2>", _reset_to_default)
            except Exception:
                pass

        # Tick marker at default position (overlayed on top of the Scale trough)
        try:
            _default_notch = int(round((float(default_value) - from_) / actual_step))
            _default_notch = max(0, min(total_steps, _default_notch))
            _pad = DEFAULT_TICK_TRACK_PAD_PCT
            _pad = max(0.0, min(0.49, float(_pad)))
            if total_steps:
                _relx = _pad + (_default_notch / float(total_steps)) * (
                    1.0 - 2.0 * _pad
                )
            else:
                _relx = 0.0
            _relx = max(0.0, min(1.0, _relx))

            _marker = tk.Frame(parent, width=DEFAULT_TICK_WIDTH, bg=DEFAULT_TICK_COLOR)
            _marker.place(
                in_=slider,
                relx=_relx,
                x=DEFAULT_TICK_X_OFFSET_PX,
                rely=DEFAULT_TICK_RELY,
                relheight=DEFAULT_TICK_RELHEIGHT,
                anchor="center",
            )
        except Exception:
            pass
    if hasattr(GUI_self, "slider_label_updaters"):
        GUI_self.slider_label_updaters.append(sync_external_change)
    if hasattr(GUI_self, "widgets_to_disable"):
        GUI_self.widgets_to_disable.append(slider)

    return lambda val: (var.set(val), sync_external_change())


def create_dual_slider_layout(
    GUI_self,
    parent: ttk.Frame,
    text_x: str,
    text_y: str,
    var_x: tk.Variable,
    var_y: tk.Variable,
    from_: float,
    to: float,
    row: int,
    decimals: int = 0,
    is_integer: bool = True,
    tooltip_key_x: Optional[str] = None,
    tooltip_key_y: Optional[str] = None,
    trough_increment: float = -1,
    display_next_odd_integer: bool = False,
    custom_label_formula: Optional[Callable] = None,  # Passed through to label display
    default_x: Optional[float] = None,
    default_y: Optional[float] = None,
    # --- NEW: Asymmetrical support ---
    from_y: Optional[float] = None,
    to_y: Optional[float] = None,
    decimals_y: Optional[int] = None,
    step_size_x: Optional[float] = None,
    step_size_y: Optional[float] = None,
) -> Tuple[ttk.Frame, Tuple[Callable, Callable], Tuple[ttk.Frame, ttk.Frame]]:
    """Creates a two-column (X/Y) slider row with optional default ticks + right-click reset."""
    xy_frame = ttk.Frame(parent)
    xy_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=0)
    xy_frame.grid_columnconfigure(0, weight=1)
    xy_frame.grid_columnconfigure(1, weight=1)

    # Use specified values or fallback to symmetrical defaults
    f_x = from_
    t_x = to
    d_x = decimals

    f_y = from_y if from_y is not None else from_
    t_y = to_y if to_y is not None else to
    d_y = decimals_y if decimals_y is not None else decimals

    x_frame = ttk.Frame(xy_frame)
    x_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    x_frame.grid_columnconfigure(1, weight=1)
    set_x = create_single_slider_with_label_updater(
        GUI_self,
        x_frame,
        text_x,
        var_x,
        f_x,
        t_x,
        0,
        decimals=d_x,
        tooltip_key=tooltip_key_x,
        trough_increment=trough_increment,
        display_next_odd_integer=display_next_odd_integer,
        custom_label_formula=custom_label_formula,
        step_size=step_size_x,
        default_value=default_x,
    )

    y_frame = ttk.Frame(xy_frame)
    y_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    y_frame.grid_columnconfigure(1, weight=1)
    set_y = create_single_slider_with_label_updater(
        GUI_self,
        y_frame,
        text_y,
        var_y,
        f_y,
        t_y,
        0,
        decimals=d_y,
        tooltip_key=tooltip_key_y,
        trough_increment=trough_increment,
        display_next_odd_integer=display_next_odd_integer,
        custom_label_formula=custom_label_formula,
        step_size=step_size_y,
        default_value=default_y,
    )
    return xy_frame, (set_x, set_y), (x_frame, y_frame)


def custom_dilate(
    tensor: torch.Tensor,
    kernel_size_x: float,
    kernel_size_y: float,
    use_gpu: bool = False,
    max_content_value: float = 1.0,
) -> torch.Tensor:
    """
    Applies 16-bit fractional dilation or erosion to preserve 10-bit+ depth fidelity.
    """
    kx_raw = float(kernel_size_x)
    ky_raw = float(kernel_size_y)

    if abs(kx_raw) <= 1e-5 and abs(ky_raw) <= 1e-5:
        return tensor

    if (kx_raw > 0 and ky_raw < 0) or (kx_raw < 0 and ky_raw > 0):
        tensor = custom_dilate(tensor, kx_raw, 0, use_gpu, max_content_value)
        return custom_dilate(tensor, 0, ky_raw, use_gpu, max_content_value)

    is_erosion = kx_raw < 0 or ky_raw < 0
    kx_abs, ky_abs = abs(kx_raw), abs(ky_raw)

    def get_dilation_params(value):
        if value <= 1e-5:
            return 1, 1, 0.0
        elif value < 3.0:
            return 1, 3, (value / 3.0)
        else:
            base = 3 + 2 * int((value - 3) // 2)
            return base, base + 2, (value - base) / 2.0

    kx_low, kx_high, tx = get_dilation_params(kx_abs)
    ky_low, ky_high, ty = get_dilation_params(ky_abs)

    device = torch.device("cpu")
    tensor_cpu = tensor.to(device)
    processed_frames = []

    for t in range(tensor_cpu.shape[0]):
        frame_float = tensor_cpu[t].numpy()
        frame_2d_raw = (
            frame_float[0]
            if frame_float.shape[0] == 1
            else np.transpose(frame_float, (1, 2, 0))
        )
        effective_max_value = max(max_content_value, 1e-5)

        # MODIFIED: Use uint16 (65535) instead of uint8 (255)
        src_img = np.ascontiguousarray(
            np.clip((frame_2d_raw / effective_max_value) * 65535, 0, 65535).astype(
                np.uint16
            )
        )

        def do_op(k_w, k_h, img):
            if k_w <= 1 and k_h <= 1:
                return img.astype(np.float32)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
            if is_erosion:
                return cv2.erode(img, kernel, iterations=1).astype(np.float32)
            return cv2.dilate(img, kernel, iterations=1).astype(np.float32)

        is_x_int, is_y_int = (tx <= 1e-4), (ty <= 1e-4)
        if is_x_int and is_y_int:
            final_float = do_op(kx_low, ky_low, src_img)
        elif not is_x_int and is_y_int:
            final_float = (1.0 - tx) * do_op(kx_low, ky_low, src_img) + tx * do_op(
                kx_high, ky_low, src_img
            )
        elif is_x_int and not is_y_int:
            final_float = (1.0 - ty) * do_op(kx_low, ky_low, src_img) + ty * do_op(
                kx_low, ky_high, src_img
            )
        else:
            r11, r12 = do_op(kx_low, ky_low, src_img), do_op(kx_low, ky_high, src_img)
            r21, r22 = do_op(kx_high, ky_low, src_img), do_op(kx_high, ky_high, src_img)
            final_float = (1.0 - tx) * ((1.0 - ty) * r11 + ty * r12) + tx * (
                (1.0 - ty) * r21 + ty * r22
            )

        # MODIFIED: Rescale back using 65535.0
        processed_raw = (final_float / 65535.0) * effective_max_value
        processed_frames.append(torch.from_numpy(processed_raw).unsqueeze(0).float())

    return torch.stack(processed_frames).to(tensor.device)


def custom_dilate_left(
    tensor: torch.Tensor,
    kernel_size: float,
    use_gpu: bool = False,
    max_content_value: float = 1.0,
) -> torch.Tensor:
    """
    Directional (one-sided) 16-bit fractional dilation to the LEFT (negative X direction in image space).
    Expands values leftward by propagating pixels from right neighbors (max filter), preserving 10-bit+ fidelity.

    kernel_size: 0 disables. Fractional values blend between two integer kernel widths, similar to custom_dilate().
    """
    k_raw = float(kernel_size)
    if abs(k_raw) <= 1e-5:
        return tensor

    # Negative values mean directional erosion (min filter) to the LEFT.
    is_erosion = k_raw < 0
    k_raw = abs(k_raw)

    # Match custom_dilate() semantics: value -> odd kernel widths with fractional blend.
    def get_dilation_params(value: float):
        if value <= 1e-5:
            return 1, 1, 0.0
        elif value < 3.0:
            return 1, 3, (value / 3.0)
        else:
            base = 3 + 2 * int((value - 3) // 2)
            return base, base + 2, (value - base) / 2.0

    k_w_low, k_w_high, t = get_dilation_params(k_raw)

    # Convert odd kernel widths (1,3,5,...) into a one-sided "reach" (radius) in pixels.
    # Standard dilation expands by radius = k_w//2 on each side; left-only uses that radius as reach.
    k_low = int(k_w_low // 2)
    k_high = int(k_w_high // 2)

    if k_low <= 0 and k_high <= 0:
        return tensor

    effective_max_value = max(float(max_content_value), 1e-5)

    device = torch.device("cpu")
    tensor = tensor.to(device)

    def do_op(k_int: int, src_img: np.ndarray) -> np.ndarray:
        if k_int <= 0:
            return src_img.astype(np.float32)

        # Width of kernel in pixels: reach is k_int pixels to the right, affecting current pixel.
        # Kernel shape: 1 x (k_int + 1) so that k_int=1 expands by 1 pixel.
        k_w = int(k_int) + 1
        kernel = np.ones((1, k_w), dtype=np.uint8)

        # Anchor at leftmost column so the kernel extends to the right: values propagate RIGHT->LEFT.
        anchor = (0, 0)

        if is_erosion:
            return cv2.erode(src_img, kernel, anchor=anchor, iterations=1).astype(
                np.float32
            )
        return cv2.dilate(src_img, kernel, anchor=anchor, iterations=1).astype(
            np.float32
        )

    processed_frames = []
    for t_idx in range(tensor.shape[0]):
        frame_float = tensor[t_idx].cpu().numpy()
        frame_2d_raw = (
            frame_float[0]
            if frame_float.shape[0] == 1
            else np.transpose(frame_float, (1, 2, 0))
        )

        frame_norm_2d = frame_2d_raw / effective_max_value
        frame_cv_uint16 = np.ascontiguousarray(
            np.clip(frame_norm_2d * 65535, 0, 65535).astype(np.uint16)
        )

        src = frame_cv_uint16.astype(np.float32)

        if abs(t) <= 1e-4:
            out = do_op(k_low, src)
        else:
            out_low = do_op(k_low, src)
            out_high = do_op(k_high, src)
            out = (1.0 - t) * out_low + t * out_high

        out_u16 = np.ascontiguousarray(np.clip(out, 0, 65535).astype(np.uint16))
        out_float = (out_u16.astype(np.float32) / 65535.0) * effective_max_value

        if frame_float.shape[0] == 1:
            processed = out_float[None, ...]
        else:
            processed = np.transpose(out_float, (2, 0, 1))

        processed_frames.append(processed)

    out_tensor = torch.from_numpy(np.stack(processed_frames, axis=0)).to(device)
    return out_tensor


def custom_blur_left_masked(
    tensor_before: torch.Tensor,
    tensor_after: torch.Tensor,
    kernel_size: int,
    use_gpu: bool = False,
    max_content_value: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies Gaussian blur ONLY to pixels that changed between tensor_before and tensor_after.
    Intended to soften the region created/affected by custom_dilate_left (or other directional ops).
    If kernel_size <= 0, returns tensor_after unchanged.
    """
    k = int(float(kernel_size))
    if k <= 0:
        return tensor_after

    # Blur expects odd kernel sizes; match existing behavior by snapping up to the next odd >= 1.
    if k % 2 == 0:
        k += 1
    k = max(k, 1)

    # If nothing changed, skip work.
    if mask is None:
        changed = (tensor_after - tensor_before).abs() > 1e-12
    else:
        changed = mask
        if changed.ndim == 3:
            changed = changed.unsqueeze(1)
        if changed.dtype != torch.bool:
            changed = changed > 0.5
        changed = changed.to(device=tensor_after.device)
    if not bool(changed.any().item()):
        return tensor_after

    blurred = custom_blur(
        tensor_after, k, k, use_gpu=use_gpu, max_content_value=max_content_value
    )

    # Apply only where changed; elsewhere keep tensor_after.
    return torch.where(changed, blurred, tensor_after)


def custom_blur(
    tensor: torch.Tensor,
    kernel_size_x: int,
    kernel_size_y: int,
    use_gpu: bool = True,
    max_content_value: float = 1.0,
) -> torch.Tensor:
    """
    Applies 16-bit Gaussian blur to prevent banding and maintain gamma accuracy.
    """
    k_x = int(kernel_size_x)
    k_y = int(kernel_size_y)
    if k_x <= 0 and k_y <= 0:
        return tensor

    k_x = k_x if k_x % 2 == 1 else k_x + 1
    k_y = k_y if k_y % 2 == 1 else k_y + 1

    device = torch.device("cpu")
    tensor = tensor.to(device)

    processed_frames = []
    for t in range(tensor.shape[0]):
        frame_float = tensor[t].cpu().numpy()
        frame_2d_raw = (
            frame_float[0]
            if frame_float.shape[0] == 1
            else np.transpose(frame_float, (1, 2, 0))
        )
        effective_max_value = max(max_content_value, 1e-5)

        # MODIFIED: Scale to 65535 for uint16 processing
        frame_norm_2d = frame_2d_raw / effective_max_value
        frame_cv_uint16 = np.ascontiguousarray(
            np.clip(frame_norm_2d * 65535, 0, 65535).astype(np.uint16)
        )

        # Apply Blur directly to 16-bit buffer
        processed_cv_uint16 = cv2.GaussianBlur(frame_cv_uint16, (k_x, k_y), 0)

        # MODIFIED: Rescale back using 65535.0
        processed_norm_float = processed_cv_uint16.astype(np.float32) / 65535.0
        processed_raw_float = processed_norm_float * effective_max_value

        blurred_tensor = torch.from_numpy(processed_raw_float).unsqueeze(0).float()
        processed_frames.append(blurred_tensor)

    return torch.stack(processed_frames).to(tensor.device)


def check_cuda_availability():
    """
    Checks if CUDA is available via PyTorch and if nvidia-smi can run.
    Sets the global CUDA_AVAILABLE flag.
    """
    global CUDA_AVAILABLE
    if torch.cuda.is_available():
        logger.info("PyTorch reports CUDA is available.")
        try:
            # Further check with nvidia-smi for robustness
            subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                check=True,
                timeout=5,
                encoding="utf-8",
            )
            logger.debug(
                "CUDA detected (nvidia-smi also ran successfully). NVENC can be used."
            )
            CUDA_AVAILABLE = True
        except FileNotFoundError:
            logger.warning(
                "nvidia-smi not found. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report."
            )
            CUDA_AVAILABLE = True  # Rely on PyTorch if nvidia-smi not found
        except subprocess.CalledProcessError:
            logger.warning(
                "nvidia-smi failed. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report."
            )
            CUDA_AVAILABLE = True  # Rely on PyTorch if nvidia-smi fails
        except subprocess.TimeoutExpired:
            logger.warning(
                "nvidia-smi check timed out. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report."
            )
            CUDA_AVAILABLE = True  # Rely on PyTorch if nvidia-smi times out
        except Exception as e:
            logger.error(
                f"Unexpected error during nvidia-smi check: {e}. Relying on PyTorch's report for CUDA."
            )
            CUDA_AVAILABLE = True  # Rely on PyTorch as a fallback
    else:
        logger.info("PyTorch reports CUDA is NOT available. NVENC will not be used.")
        CUDA_AVAILABLE = False
    return CUDA_AVAILABLE


def draw_progress_bar(current, total, bar_length=50, prefix="Progress:", suffix=""):
    """
    Draws an ASCII progress bar in the console, overwriting the same line.
    Adds a newline only when 100% complete. This uses `print` for direct console output.
    """
    if total == 0:
        print(f"\r{prefix} [Skipped (Total 0)] {suffix}", end="")
        return

    percent = 100 * (current / float(total))
    filled_length = int(round(bar_length * current / float(total)))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    # Format the suffix for completion
    actual_suffix = suffix
    if current == total:
        actual_suffix = "Complete"

    print(f"\r{prefix} |{bar}| {percent:.1f}% {actual_suffix}", end="", flush=True)

    if current == total:
        print()  # Add a final newline when done


def encode_frames_to_mp4(
    temp_png_dir: str,
    final_output_mp4_path: str,
    fps: float,
    total_output_frames: int,
    video_stream_info: Optional[dict],
    stop_event: Optional[threading.Event] = None,
    sidecar_json_data: Optional[dict] = None,
    user_output_crf: Optional[int] = None,  # NEW: Add this parameter
    output_sidecar_ext: str = ".json",
) -> bool:
    """
    Encodes a sequence of 16-bit PNG frames from a temporary directory into an MP4 video
    using FFmpeg, attempting to preserve color metadata and using NVENC if available.
    Also creates a sidecar JSON file if sidecar_json_data is provided.
    Returns True on success, False on failure or stop.
    """
    if total_output_frames == 0:
        logger.warning(
            f"No frames to encode for {os.path.basename(final_output_mp4_path)}. Skipping encoding."
        )
        if os.path.exists(temp_png_dir):
            shutil.rmtree(temp_png_dir)
        return False

    logger.debug(
        f"Starting FFmpeg encoding from PNG sequence to {os.path.basename(final_output_mp4_path)}"
    )
    logger.debug(f"Input PNG directory: {temp_png_dir}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",  # Overwrite output files without asking
        "-framerate",
        str(fps),  # Input framerate for the PNG sequence
        "-i",
        os.path.join(temp_png_dir, "%05d.png"),  # Input PNG sequence pattern
    ]

    # --- Determine Output Codec, Bit-Depth, and Quality ---
    output_codec = "libx264"  # Default to H.264 CPU encoder
    output_pix_fmt = "yuv420p"  # Default to 8-bit
    default_cpu_crf = "23"  # Default CRF for H.264 (lower is better quality)
    output_profile = "main"
    x265_params = []  # For specific x265 parameters

    nvenc_preset = "medium"  # Default NVENC preset (e.g., fast, medium, slow, quality)
    default_nvenc_cq = (
        "23"  # Constant Quality value for NVENC (lower is better quality)
    )

    # NEW: Apply user-specified CRF if provided
    if user_output_crf is not None and user_output_crf >= 0:
        logger.debug(f"Using user-specified output CRF: {user_output_crf}")
        default_cpu_crf = str(user_output_crf)
        default_nvenc_cq = str(
            user_output_crf
        )  # Assume user CRF applies to NVENC CQ as well for simplicity
    else:
        logger.debug("Using auto-determined output CRF.")

    is_hdr_source = False
    original_codec_name = (
        video_stream_info.get("codec_name") if video_stream_info else None
    )
    original_pix_fmt = video_stream_info.get("pix_fmt") if video_stream_info else None

    if video_stream_info:
        if (
            video_stream_info.get("color_primaries") == "bt2020"
            and video_stream_info.get("transfer_characteristics") == "smpte2084"
        ):
            is_hdr_source = True
            logger.debug("Detected HDR source. Targeting HEVC 10-bit HDR output.")

    is_original_10bit_or_higher = False
    if original_pix_fmt:
        if (
            "10" in original_pix_fmt
            or "12" in original_pix_fmt
            or "16" in original_pix_fmt
        ):
            is_original_10bit_or_higher = True

    if is_hdr_source:
        output_codec = "libx265"
        if CUDA_AVAILABLE:
            output_codec = "hevc_nvenc"
            logger.debug("    (Using hevc_nvenc for hardware acceleration)")
        output_pix_fmt = "yuv420p10le"
        if user_output_crf is None:
            default_cpu_crf = (
                "28"  # For CPU x265 (HDR often needs higher CRF to look "good")
            )
        output_profile = "main10"
        if video_stream_info.get("mastering_display_metadata"):
            x265_params.append(
                f"master-display={video_stream_info['mastering_display_metadata']}"
            )
        if video_stream_info.get("max_content_light_level"):
            x265_params.append(
                f"max-cll={video_stream_info['max_content_light_level']}"
            )
    elif original_codec_name == "hevc" and is_original_10bit_or_higher:
        logger.debug(
            "Detected SDR 10-bit HEVC source. Targeting HEVC 10-bit SDR output."
        )
        output_codec = "libx265"
        if CUDA_AVAILABLE:
            output_codec = "hevc_nvenc"
            logger.debug("    (Using hevc_nvenc for hardware acceleration)")
        output_pix_fmt = "yuv420p10le"
        if user_output_crf is None:
            default_cpu_crf = "24"  # For CPU x265 (SDR 10-bit)
        output_profile = "main10"
    else:  # Default to H.264 8-bit, or if no info
        logger.debug(
            "Detected SDR (8-bit H.264 or other) source or no specific info. Targeting H.264 8-bit."
        )
        output_codec = "libx264"
        if CUDA_AVAILABLE:
            output_codec = "h264_nvenc"
            logger.debug("    (Using h264_nvenc for hardware acceleration)")
        output_pix_fmt = "yuv420p"
        if user_output_crf is None:
            default_cpu_crf = "18"  # For CPU x264 (SDR 8-bit, higher quality)
        output_profile = "main"

    logger.debug(f"default_cpu_crf = {default_cpu_crf}")
    # Add codec, profile, pix_fmt
    ffmpeg_cmd.extend(["-c:v", output_codec])
    if "nvenc" in output_codec:
        ffmpeg_cmd.extend(["-preset", nvenc_preset])
        ffmpeg_cmd.extend(["-cq", default_nvenc_cq])  # NVENC uses CQ, not CRF
    else:
        ffmpeg_cmd.extend(["-crf", default_cpu_crf])

    ffmpeg_cmd.extend(["-pix_fmt", output_pix_fmt])
    if output_profile:
        ffmpeg_cmd.extend(["-profile:v", output_profile])

    # Add x265-params if using libx265 and params are available
    if output_codec == "libx265" and x265_params:
        ffmpeg_cmd.extend(["-x265-params", ":".join(x265_params)])

    # Add general color flags if present in source info
    if video_stream_info:
        if video_stream_info.get("color_primaries"):
            ffmpeg_cmd.extend(
                ["-color_primaries", video_stream_info["color_primaries"]]
            )
        if video_stream_info.get("transfer_characteristics"):
            ffmpeg_cmd.extend(
                ["-color_trc", video_stream_info["transfer_characteristics"]]
            )
        if video_stream_info.get("color_space"):
            ffmpeg_cmd.extend(["-colorspace", video_stream_info["color_space"]])
        # Ensure color range metadata is tagged when available (metadata-only)
        if video_stream_info.get("color_range") in ("tv", "pc"):
            ffmpeg_cmd.extend(["-color_range", video_stream_info["color_range"]])

    # Final output path
    # Write color info into the container (MP4/MOV) so tags survive in downstream tools
    _ext = os.path.splitext(final_output_mp4_path)[1].lower()
    if _ext in (".mp4", ".mov", ".m4v"):
        ffmpeg_cmd.extend(["-movflags", "+write_colr"])

    ffmpeg_cmd.append(final_output_mp4_path)
    logger.debug(f"FFmpeg command: {' '.join(ffmpeg_cmd)}")
    process = None

    # --- NEW: Helper to read FFmpeg's output without blocking ---
    def _read_ffmpeg_output(pipe, log_level):
        try:
            # Use iter to read line by line, which is non-blocking
            for line in iter(pipe.readline, ""):
                if line:
                    logger.log(log_level, f"FFmpeg: {line.strip()}")
        except Exception as e:
            logger.error(f"Error reading FFmpeg pipe: {e}")
        finally:
            if pipe:
                pipe.close()

    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )

        # --- NEW: Start threads to read stdout and stderr to prevent deadlock ---
        stdout_thread = threading.Thread(
            target=_read_ffmpeg_output,
            args=(process.stdout, logging.DEBUG),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_read_ffmpeg_output, args=(process.stderr, logging.INFO), daemon=True
        )
        stdout_thread.start()
        stderr_thread.start()

        while process.poll() is None:  # While process is still running
            if stop_event and stop_event.is_set():
                logger.warning(
                    f"FFmpeg encoding stopped by user for {os.path.basename(final_output_mp4_path)}."
                )
                process.terminate()  # or process.kill()
                process.wait(timeout=5)
                return False
            time.sleep(0.1)  # Check stop_event frequently

        # Wait for the process and reader threads to complete
        process.wait(timeout=120)
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

        if process.returncode != 0:
            logger.error(
                f"FFmpeg encoding failed for {os.path.basename(final_output_mp4_path)} (return code {process.returncode}). Check console for FFmpeg output."
            )
            return False
        else:
            logger.debug(f"Successfully encoded video to {final_output_mp4_path}")

    except FileNotFoundError:
        logger.error(
            "FFmpeg not found. Please ensure FFmpeg is installed and in your system PATH."
        )
        return False
    except subprocess.CalledProcessError as e:
        logger.error(
            f"FFmpeg encoding failed for {os.path.basename(final_output_mp4_path)}: {e.stderr}\n{e.stdout}"
        )
        return False
    except subprocess.TimeoutExpired as e:
        logger.error(
            f"FFmpeg encoding timed out for {os.path.basename(final_output_mp4_path)}: {e.stderr}"
        )
        process.kill()
        process.wait()  # Ensure the process is cleaned up
        return False
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during encoding for {os.path.basename(final_output_mp4_path)}: {str(e)}",
            exc_info=True,
        )
        return False
    finally:
        # Cleanup temporary PNGs
        if os.path.exists(temp_png_dir):
            try:
                shutil.rmtree(temp_png_dir)
                logger.debug(f"Cleaned up temporary directory: {temp_png_dir}")
            except Exception as e:
                logger.error(
                    f"Error cleaning up temporary PNG directory {temp_png_dir}: {e}"
                )

    # Write sidecar JSON if data is provided
    if sidecar_json_data:
        output_sidecar_path = (
            f"{os.path.splitext(final_output_mp4_path)[0]}{output_sidecar_ext}"
        )
        try:
            with open(output_sidecar_path, "w", encoding="utf-8") as f:
                json.dump(sidecar_json_data, f, indent=4)
            logger.info(f"Created output sidecar file: {output_sidecar_path}")
        except Exception as e:
            logger.error(
                f"Error creating output sidecar file '{output_sidecar_path}': {e}"
            )
            # This is not a critical error for video encoding, so don't return False here.

    logger.info(f"Done processing {os.path.basename(final_output_mp4_path)}")
    return True


def get_video_stream_info(video_path: str) -> Optional[dict]:
    """
    Extracts comprehensive video stream metadata using ffprobe.
    Returns a dict with relevant color properties, codec, pixel format, and HDR mastering metadata
    or None if ffprobe fails/info not found.
    Requires ffprobe to be installed and in your system PATH.
    This function *does not* show messageboxes; the caller should handle errors.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",  # Select the first video stream
        "-show_entries",
        "stream=codec_name,profile,pix_fmt,color_range,color_primaries,transfer_characteristics,color_space,r_frame_rate",
        "-show_entries",
        "side_data=mastering_display_metadata,max_content_light_level",  # ADDED entries
        "-of",
        "json",
        video_path,
    ]

    try:
        # Check if ffprobe is available without showing a messagebox
        subprocess.run(
            ["ffprobe", "-version"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=10,
        )
    except FileNotFoundError:
        logger.error(
            "ffprobe not found. Please ensure FFmpeg is installed and in your system PATH."
        )
        return None
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running ffprobe check: {e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        logger.error("ffprobe check timed out.")
        return None

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
            timeout=500,
        )
        data = json.loads(result.stdout)

        stream_info = {}
        if "streams" in data and len(data["streams"]) > 0:
            s = data["streams"][0]
            # Common video stream properties
            for key in [
                "codec_name",
                "profile",
                "pix_fmt",
                "color_range",
                "color_primaries",
                "transfer_characteristics",
                "color_space",
                "r_frame_rate",
            ]:
                if key in s:
                    stream_info[key] = s[key]

            # HDR mastering display and CLL metadata (often in side_data_list, but sometimes also directly in stream)
            # Prioritize stream-level if available, otherwise check side_data_list
            if "mastering_display_metadata" in s:
                stream_info["mastering_display_metadata"] = s[
                    "mastering_display_metadata"
                ]
            if "max_content_light_level" in s:
                stream_info["max_content_light_level"] = s["max_content_light_level"]

        # Check side_data_list if stream-level properties weren't found or for additional data
        if "side_data_list" in data:
            for sd in data["side_data_list"]:
                if (
                    "mastering_display_metadata" in sd
                    and "mastering_display_metadata" not in stream_info
                ):
                    stream_info["mastering_display_metadata"] = sd[
                        "mastering_display_metadata"
                    ]
                if (
                    "max_content_light_level" in sd
                    and "max_content_light_level" not in stream_info
                ):
                    stream_info["max_content_light_level"] = sd[
                        "max_content_light_level"
                    ]

        # Filter out empty strings/None/N/A values
        filtered_info = {
            k: v
            for k, v in stream_info.items()
            if v and v not in ["N/A", "und", "unknown"]
        }
        return filtered_info if filtered_info else None

    except subprocess.CalledProcessError as e:
        logger.error(
            f"ffprobe failed for {video_path} (return code {e.returncode}):\n{e.stderr}"
        )
        return None
    except subprocess.TimeoutExpired:
        logger.error(f"ffprobe timed out for {video_path}.")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse ffprobe output for {video_path}: {e}")
        logger.debug(f"Raw ffprobe stdout: {result.stdout}")
        return None
    except Exception as e:
        logger.error(
            f"An unexpected error occurred with ffprobe for {video_path}: {e}",
            exc_info=True,
        )
        return None


def release_cuda_memory():
    """Releases GPU memory and performs garbage collection."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared.")
        gc.collect()
        logger.debug("Python garbage collector invoked.")
    except Exception as e:
        logger.error(
            f"Error releasing VRAM or during garbage collection: {e}", exc_info=True
        )


def read_video_frames_decord(
    video_path: str,
    process_length: int = -1,
    target_fps: float = -1.0,
    set_res_width: Optional[int] = None,
    set_res_height: Optional[int] = None,
    decord_ctx=cpu(0),
) -> Tuple[np.ndarray, float, int, int, int, int, Optional[dict]]:
    """
    Reads video frames using decord, optionally resizing and downsampling frame rate.
    Returns frames as a 4D float32 numpy array [T, H, W, C] normalized to 0-1,
    the actual output FPS, original video height/width, actual processed height/width,
    and video stream metadata.
    """
    logger.info(f"Reading video: {os.path.basename(video_path)}")

    # Get video stream info first for FPS detection
    video_stream_info = get_video_stream_info(video_path)

    # Use a dummy VideoReader to get original dimensions without loading all frames
    temp_reader = VideoReader(video_path, ctx=cpu(0))
    original_height, original_width = temp_reader.get_batch([0]).shape[1:3]
    del temp_reader  # Release immediately

    height_for_decord = original_height
    width_for_decord = original_width

    if (
        set_res_width is not None
        and set_res_width > 0
        and set_res_height is not None
        and set_res_height > 0
    ):
        height_for_decord = set_res_height
        width_for_decord = set_res_width
        logger.info(
            f"Targeting specific resolution for decord: {width_for_decord}x{height_for_decord}"
        )
    else:
        logger.info(
            f"Using original video resolution for decord: {original_width}x{original_height}"
        )

    # Initialize VideoReader with potential target resolution
    vid_reader = VideoReader(
        video_path, ctx=decord_ctx, width=width_for_decord, height=height_for_decord
    )
    num_total_frames = len(vid_reader)

    if num_total_frames == 0:
        logger.warning(f"No frames found in {video_path}.")
        return (
            np.empty((0, 0, 0, 0), dtype=np.float32),
            0.0,
            original_height,
            original_width,
            0,
            0,
            video_stream_info,
        )

    # Determine FPS: Use ffprobe's r_frame_rate if reliable, otherwise decord's avg_fps, or target_fps
    actual_output_fps = 0.0
    if target_fps != -1.0 and target_fps > 0:
        actual_output_fps = target_fps
        logger.info(f"Using user-specified target FPS: {actual_output_fps:.2f}")
    elif video_stream_info and "r_frame_rate" in video_stream_info:
        try:
            r_frame_rate_str = video_stream_info["r_frame_rate"].split("/")
            if len(r_frame_rate_str) == 2:
                actual_output_fps = float(r_frame_rate_str[0]) / float(
                    r_frame_rate_str[1]
                )
            else:
                actual_output_fps = float(r_frame_rate_str[0])
            logger.info(
                f"Using ffprobe FPS: {actual_output_fps:.2f} for {os.path.basename(video_path)}"
            )
        except (ValueError, ZeroDivisionError):
            actual_output_fps = vid_reader.get_avg_fps()
            logger.warning(
                f"Failed to parse ffprobe FPS. Falling back to Decord avg_fps: {actual_output_fps:.2f}"
            )
    else:
        actual_output_fps = vid_reader.get_avg_fps()
        logger.info(
            f"Using Decord avg_fps: {actual_output_fps:.2f} for {os.path.basename(video_path)}"
        )

    stride = max(round(vid_reader.get_avg_fps() / actual_output_fps), 1)
    frames_idx = list(range(0, num_total_frames, stride))

    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
        logger.info(
            f"Limiting to {len(frames_idx)} frames based on process_length parameter."
        )

    if not frames_idx:
        logger.warning(
            "No frames selected for processing after stride and process_length filters."
        )
        return (
            np.empty((0, 0, 0, 0), dtype=np.float32),
            0.0,
            original_height,
            original_width,
            0,
            0,
            video_stream_info,
        )

    frames_batch = vid_reader.get_batch(frames_idx)
    frames_numpy = (
        frames_batch.asnumpy().astype("float32") / 255.0
    )  # Normalize to 0-1 float32

    # Get actual processed height/width after Decord (might differ from target if source is smaller)
    actual_processed_height, actual_processed_width = frames_numpy.shape[1:3]
    logger.info(
        f"Read {len(frames_idx)} frames. Original: {original_width}x{original_height}, Processed: {actual_processed_width}x{actual_processed_height}"
    )

    return (
        frames_numpy,
        actual_output_fps,
        original_height,
        original_width,
        actual_processed_height,
        actual_processed_width,
        video_stream_info,
    )


def set_util_logger_level(level):
    """Sets the logging level for the 'stereocrafter_util' logger."""
    logger.setLevel(level)
    # If basicConfig was already called, its handlers might not update automatically.
    # Ensure handlers also reflect the new level.
    for handler in logger.handlers:
        handler.setLevel(level)


def start_ffmpeg_pipe_process(
    content_width: int,
    content_height: int,
    final_output_mp4_path: str,
    fps: float,
    video_stream_info: Optional[dict],
    output_format_str: str = "",  # Make argument optional with a default value
    user_output_crf: Optional[int] = None,
    pad_to_16_9: bool = False,
    debug_label: Optional[str] = None,
) -> Optional[subprocess.Popen]:
    """
    Builds an FFmpeg command and starts a subprocess configured to accept
    raw 16-bit BGR video frames from stdin.

    If pad_to_16_9 is True, it will letterbox the output to a 16:9 aspect ratio.

    Returns the Popen object on success, None on failure.
    """
    logger.debug(
        f"Starting FFmpeg pipe process for {os.path.basename(final_output_mp4_path)}"
    )

    # --- NEW: Padding Logic ---
    vf_options = []
    output_width = content_width
    output_height = content_height

    if pad_to_16_9:
        # --- FIX: Calculate padding based on single-eye width ---
        # Determine the width of a single eye based on the output format
        if output_format_str in [
            "Full SBS (Left-Right)",
            "Full SBS Cross-eye (Right-Left)",
            "Double SBS",
        ]:
            single_eye_width = content_width // 2
        else:  # Half SBS, Anaglyph, Right-Eye Only
            single_eye_width = content_width

        # Calculate the target 16:9 height based on the single eye's width
        target_16_9_height = int(single_eye_width * 9 / 16)
        # Ensure the height is an even number for codec compatibility
        if target_16_9_height % 2 != 0:
            target_16_9_height += 1

        if target_16_9_height > content_height:
            output_height = target_16_9_height
            # The output width for padding is always the full content width
            vf_options.append(
                f"pad=w={output_width}:h={output_height}:x=0:y=(oh-ih)/2:color=black"
            )
            logger.debug(
                f"Padding enabled. Content: {content_width}x{content_height}, Container: {output_width}x{output_height}"
            )

    # --- This command-building logic is adapted from the original encode_frames_to_mp4 ---
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{content_width}x{content_height}",  # Input pipe is always the content size
        "-pix_fmt",
        "bgr48le",  # Input is 16-bit BGR from OpenCV
        "-r",
        str(fps),
        "-i",
        "-",  # Read input from stdin pipe
    ]

    # --- Determine Output Codec, Bit-Depth, and Quality ---
    output_codec = "libx264"
    output_pix_fmt = "yuv420p"
    default_cpu_crf = "23"
    output_profile = "main"
    x265_params = []
    nvenc_preset = "medium"
    default_nvenc_cq = "23"

    if user_output_crf is not None and user_output_crf >= 0:
        logger.debug(f"Using user-specified output CRF/CQ: {user_output_crf}")
        default_cpu_crf = str(user_output_crf)
        default_nvenc_cq = str(user_output_crf)
    else:
        logger.debug("Using auto-determined output CRF.")

    is_hdr_source = False
    original_codec_name = (
        video_stream_info.get("codec_name") if video_stream_info else None
    )
    original_pix_fmt = video_stream_info.get("pix_fmt") if video_stream_info else None

    if video_stream_info:
        if (
            video_stream_info.get("color_primaries") == "bt2020"
            and video_stream_info.get("transfer_characteristics") == "smpte2084"
        ):
            is_hdr_source = True
            logger.debug("Detected HDR source. Targeting HEVC 10-bit HDR output.")

    is_original_10bit_or_higher = False
    if original_pix_fmt:
        if (
            "10" in original_pix_fmt
            or "12" in original_pix_fmt
            or "16" in original_pix_fmt
        ):
            is_original_10bit_or_higher = True

    if is_hdr_source:
        output_codec = "libx265"
        if CUDA_AVAILABLE:
            output_codec = "hevc_nvenc"
        output_pix_fmt = "yuv420p10le"
        if user_output_crf is None:
            default_cpu_crf = "28"
        output_profile = "main10"
        if video_stream_info.get("mastering_display_metadata"):
            x265_params.append(
                f"master-display={video_stream_info['mastering_display_metadata']}"
            )
        if video_stream_info.get("max_content_light_level"):
            x265_params.append(
                f"max-cll={video_stream_info['max_content_light_level']}"
            )
    elif original_codec_name == "hevc" and is_original_10bit_or_higher:
        output_codec = "libx265"
        if CUDA_AVAILABLE:
            output_codec = "hevc_nvenc"
        output_pix_fmt = "yuv420p10le"
        if user_output_crf is None:
            default_cpu_crf = "24"
        output_profile = "main10"
    else:
        output_codec = "libx264"
        if CUDA_AVAILABLE:
            output_codec = "h264_nvenc"
        output_pix_fmt = "yuv420p"
        if user_output_crf is None:
            default_cpu_crf = "18"
        output_profile = "main"

    ffmpeg_cmd.extend(["-c:v", output_codec])
    if "nvenc" in output_codec:
        ffmpeg_cmd.extend(["-preset", nvenc_preset, "-qp", default_nvenc_cq])
    else:
        ffmpeg_cmd.extend(["-crf", default_cpu_crf])

    ffmpeg_cmd.extend(["-pix_fmt", output_pix_fmt])
    if output_profile:
        ffmpeg_cmd.extend(["-profile:v", output_profile])

    if output_codec == "libx265" and x265_params:
        ffmpeg_cmd.extend(["-x265-params", ":".join(x265_params)])

    # --- MODIFIED: Add default color space tags for robustness ---
    # Use a dictionary's .get() with a default value to prevent errors if tags are missing.
    # The most common standard for SDR HD video is BT.709.
    color_primaries = (
        video_stream_info.get("color_primaries", "bt709")
        if video_stream_info is not None
        else "bt709"
    )
    transfer_characteristics = (
        video_stream_info.get("transfer_characteristics", "bt709")
        if video_stream_info is not None
        else "bt709"
    )
    color_space = (
        video_stream_info.get("color_space", "bt709")
        if video_stream_info is not None
        else "bt709"
    )

    # --- DEBUG: Dump ffprobe-derived color metadata + encoding flags (Hi/Lo parity checks) ---
    # Non-invasive: only logs + tags the spawned process with a dict.
    try:
        src_pix_fmt = video_stream_info.get("pix_fmt") if video_stream_info else None
        src_range = video_stream_info.get("color_range") if video_stream_info else None
        src_prim = (
            video_stream_info.get("color_primaries") if video_stream_info else None
        )
        src_trc = (
            video_stream_info.get("transfer_characteristics")
            if video_stream_info
            else None
        )
        src_matrix = video_stream_info.get("color_space") if video_stream_info else None
    except Exception:
        src_pix_fmt = src_range = src_prim = src_trc = src_matrix = None

    quality_mode = "qp" if "nvenc" in output_codec else "crf"
    quality_value = default_nvenc_cq if "nvenc" in output_codec else default_cpu_crf

    sc_encode_flags = {
        "enc_codec": output_codec,
        "enc_pix_fmt": output_pix_fmt,
        "enc_profile": output_profile,
        "enc_color_primaries": color_primaries,
        "enc_color_trc": transfer_characteristics,
        "enc_colorspace": color_space,
        # NOTE: we currently do not set -color_range explicitly on output; record the source for debugging.
        "src_pix_fmt": src_pix_fmt,
        "src_color_range": src_range,
        "src_color_primaries": src_prim,
        "src_color_trc": src_trc,
        "src_colorspace": src_matrix,
        "quality_mode": quality_mode,
        "quality_value": quality_value,
    }

    if debug_label:
        logger.info(
            f"[COLOR_META][{debug_label}] src(pix_fmt={src_pix_fmt}, range={src_range}, primaries={src_prim}, trc={src_trc}, matrix={src_matrix}) "
            f"-> enc(codec={output_codec}, pix_fmt={output_pix_fmt}, profile={output_profile}, primaries={color_primaries}, trc={transfer_characteristics}, matrix={color_space}, {quality_mode}={quality_value})"
        )
    # --- END DEBUG ---

    # Add the determined or default flags to the command
    ffmpeg_cmd.extend(["-color_primaries", color_primaries])
    ffmpeg_cmd.extend(["-color_trc", transfer_characteristics])
    ffmpeg_cmd.extend(["-colorspace", color_space])
    # Ensure color range metadata is tagged when available (metadata-only)
    if video_stream_info and video_stream_info.get("color_range") in ("tv", "pc"):
        ffmpeg_cmd.extend(["-color_range", video_stream_info["color_range"]])
    # --- END MODIFICATION ---

    # --- NEW: Add video filters if any are defined ---
    if vf_options:
        ffmpeg_cmd.extend(["-vf", ",".join(vf_options)])
    # --- END NEW ---

    # Write color info into the container (MP4/MOV) so tags survive in downstream tools
    _ext = os.path.splitext(final_output_mp4_path)[1].lower()
    if _ext in (".mp4", ".mov", ".m4v"):
        ffmpeg_cmd.extend(["-movflags", "+write_colr"])

    ffmpeg_cmd.append(final_output_mp4_path)
    logger.debug(f"FFmpeg pipe command: {' '.join(ffmpeg_cmd)}")

    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            process.sc_encode_flags = sc_encode_flags  # type: ignore[attr-defined]
        except Exception:
            pass
        return process
    except FileNotFoundError:
        logger.error(
            "FFmpeg not found. Please ensure FFmpeg is installed and in your system PATH."
        )
        return None
    except Exception as e:
        logger.error(f"Failed to start FFmpeg pipe process: {e}", exc_info=True)
        return None
