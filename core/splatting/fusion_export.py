"""Fusion export sidecar generation module.

Handles parsing Fusion Export files, matching them to depth maps,
and generating/saving FSSIDECAR files using carry-forward logic.
"""

import glob
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # Fallback/stub for systems without moviepy
    class VideoFileClip:
        """Stub for systems without moviepy.

        Provides a minimal interface compatible with the moviepy VideoFileClip
        for systems where moviepy is not installed.
        """

        def __init__(self, *args, **kwargs):
            """Initialize the stub VideoFileClip.

            Args:
                *args: Variable arguments (ignored)
                **kwargs: Keyword arguments (ignored)
            """
            pass

        def close(self):
            """Close the video file (no-op for stub)."""
            pass

        @property
        def fps(self):
            """Get frames per second (returns None for stub)."""
            return None

        @property
        def duration(self):
            """Get video duration in seconds (returns None for stub)."""
            return None


logger = logging.getLogger(__name__)


class FusionSidecarGenerator:
    """
    Handles parsing Fusion Export files and generating sidecar files.

    This class processes Fusion (.fsexport) files which contain
    metadata markers with depth/stereo parameters. It matches these
    markers to depth map videos and generates sidecar files with
    carry-forward logic for parameter inheritance.

    Args:
        master_gui: Reference to main GUI for status updates
        sidecar_manager: SidecarConfigManager instance for file operations

    Example:
        >>> from dependency.stereocrafter_util import SidecarConfigManager
        >>> from core.splatting.fusion_export import FusionSidecarGenerator
        >>>
        >>> sidecar_manager = SidecarConfigManager()
        >>> generator = FusionSidecarGenerator(main_gui, sidecar_manager)
        >>> generator.generate_sidecars()  # Opens file dialogs
    """

    # Configuration mapping Fusion export keys to sidecar keys
    FUSION_PARAMETER_CONFIG = {
        "convergence": {
            "label": "Convergence Plane",
            "type": float,
            "default": 0.5,
            "fusion_key": "Convergence",
            "sidecar_key": "convergence_plane",
            "decimals": 3,
        },
        "max_disparity": {
            "label": "Max Disparity",
            "type": float,
            "default": 35.0,
            "fusion_key": "MaxDisparity",
            "sidecar_key": "max_disparity",
            "decimals": 1,
        },
        "gamma": {
            "label": "Gamma Correction",
            "type": float,
            "default": 1.0,
            "fusion_key": "FrontGamma",
            "sidecar_key": "gamma",
            "decimals": 2,
        },
        "frame_overlap": {
            "label": "Frame Overlap",
            "type": float,
            "default": 3,
            "fusion_key": "Overlap",
            "sidecar_key": "frame_overlap",
            "decimals": 0,
        },
        "input_bias": {
            "label": "Input Bias",
            "type": float,
            "default": 0.0,
            "fusion_key": "Bias",
            "sidecar_key": "input_bias",
            "decimals": 2,
        },
        "left_border": {
            "label": "Left Border",
            "type": float,
            "default": 0.0,
            "fusion_key": "LeftBorder",
            "sidecar_key": "left_border",
            "decimals": 3,
        },
        "right_border": {
            "label": "Right Border",
            "type": float,
            "default": 0.0,
            "fusion_key": "RightBorder",
            "sidecar_key": "right_border",
            "decimals": 3,
        },
        "manual_border": {
            "label": "Border Mode",
            "type": str,
            "default": "Off",
            "fusion_key": "BorderMode",
            "sidecar_key": "border_mode",
            "decimals": 0,
        },
        "auto_border_l": {
            "label": "Auto Border L",
            "type": float,
            "default": 0.0,
            "fusion_key": "AutoBorderL",
            "sidecar_key": "auto_border_L",
            "decimals": 3,
        },
        "auto_border_r": {
            "label": "Auto Border R",
            "type": float,
            "default": 0.0,
            "fusion_key": "AutoBorderR",
            "sidecar_key": "auto_border_R",
            "decimals": 3,
        },
    }

    def __init__(self, master_gui, sidecar_manager):
        """Initialize the Fusion sidecar generator.

        Args:
            master_gui: GUI object with status_label and other UI elements
            sidecar_manager: SidecarConfigManager for file operations
        """
        self.master_gui = master_gui
        self.sidecar_manager = sidecar_manager
        self.logger = logging.getLogger(__name__)

    def _get_video_frame_count(self, file_path: str) -> int:
        """Safely get the frame count of a video file using moviepy.

        Args:
            file_path: Path to the video file

        Returns:
            Number of frames, or 0 if determination fails
        """
        try:
            clip = VideoFileClip(file_path)
            fps = clip.fps
            duration = clip.duration
            if fps is None or duration is None:
                fps = 24
                if duration is None:
                    return 0

            frames = math.ceil(duration * fps)
            clip.close()
            return frames
        except Exception as e:
            self.logger.warning(
                f"Error getting frame count for {os.path.basename(file_path)}: {e}"
            )
            return 0

    def _load_and_validate_fsexport(self, file_path: str) -> Optional[List[Dict]]:
        """Load, parse, and validate marker data from a Fusion Export file.

        Args:
            file_path: Path to the .fsexport file

        Returns:
            List of marker dictionaries, or None if invalid
        """
        try:
            with open(file_path, "r") as f:
                export_data = json.load(f)
        except json.JSONDecodeError as e:
            self.logger.error(
                f"Failed to parse JSON in {os.path.basename(file_path)}: {e}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Failed to read {os.path.basename(file_path)}: {e}"
            )
            return None

        markers = export_data.get("markers", [])
        if not markers:
            self.logger.warning("No 'markers' found in the export file.")
            return None

        # Sort markers by frame number (critical for carry-forward logic)
        markers.sort(key=lambda m: m["frame"])
        self.logger.info(
            f"Loaded {len(markers)} markers from {os.path.basename(file_path)}."
        )
        return markers

    def _scan_target_videos(
        self, folder: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Scan a folder for video files and compute their frame counts.

        Args:
            folder: Path to folder containing depth map videos

        Returns:
            List of video data dictionaries, or None if no videos found
        """
        video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        found_files_paths = []
        for ext in video_extensions:
            found_files_paths.extend(glob.glob(os.path.join(folder, ext)))
        sorted_files_paths = sorted(found_files_paths)

        if not sorted_files_paths:
            self.logger.warning(f"No video depth map files found in: {folder}")
            return None

        target_video_data = []
        cumulative_frames = 0

        for full_path in sorted_files_paths:
            total_frames = self._get_video_frame_count(full_path)

            if total_frames == 0:
                self.logger.warning(
                    f"Skipping {os.path.basename(full_path)} due to zero frame count."
                )
                continue

            target_video_data.append(
                {
                    "full_path": full_path,
                    "basename": os.path.basename(full_path),
                    "total_frames": total_frames,
                    "timeline_start_frame": cumulative_frames,
                    "timeline_end_frame": cumulative_frames + total_frames - 1,
                }
            )
            cumulative_frames += total_frames

        self.logger.info(
            f"Scanned {len(target_video_data)} video files. "
            f"Total timeline frames: {cumulative_frames}."
        )
        return target_video_data

    def _apply_parameters(
        self,
        target_videos: List[Dict[str, Any]],
        markers: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], int]:
        """Apply parameters from markers to videos using carry-forward logic.

        Args:
            target_videos: List of video data dictionaries
            markers: List of marker dictionaries from Fusion export

        Returns:
            Tuple of (sidecar_data, applied_count)
        """
        applied_count = 0

        # Initialize last known values with defaults
        last_param_vals = {}
        for key, config in self.FUSION_PARAMETER_CONFIG.items():
            last_param_vals[key] = config["default"]

        for file_data in target_videos:
            file_start_frame = file_data["timeline_start_frame"]

            # Find most relevant marker (latest <= file_start_frame)
            relevant_marker = None
            for marker in markers:
                if marker["frame"] <= file_start_frame:
                    relevant_marker = marker
                else:
                    break

            current_param_vals = last_param_vals.copy()

            if relevant_marker and relevant_marker.get("values"):
                marker_values = relevant_marker["values"]
                updated_from_marker = False

                for key, config in self.FUSION_PARAMETER_CONFIG.items():
                    fusion_key = config["fusion_key"]
                    default_val = config["default"]

                    if fusion_key in marker_values:
                        val = marker_values.get(fusion_key, default_val)
                        try:
                            current_param_vals[key] = config["type"](val)
                            updated_from_marker = True
                        except (ValueError, TypeError):
                            self.logger.warning(
                                f"Marker value for '{fusion_key}' is invalid ({val}). "
                                "Using previous/default value."
                            )

                if updated_from_marker:
                    applied_count += 1

            # Update for carry-forward
            last_param_vals = current_param_vals.copy()

        # Build sidecar data from final parameter values
        sidecar_data = {}
        for key, config in self.FUSION_PARAMETER_CONFIG.items():
            value = last_param_vals[key]
            if config["type"] is bool:
                sidecar_data[config["sidecar_key"]] = bool(value)
            elif config["type"] is str:
                # String values (like border_mode) don't need rounding
                sidecar_data[config["sidecar_key"]] = str(value)
            else:
                sidecar_data[config["sidecar_key"]] = round(
                    float(value), config["decimals"]
                )

        return sidecar_data, applied_count

    def generate_sidecars(self, filedialog=None, messagebox=None) -> None:
        """Main entry point for Fusion Export to Sidecar generation workflow.

        Opens file dialogs to select the export file and target folder,
        then generates sidecar files for matching depth maps.

        Args:
            filedialog: tkinter filedialog module (optional, defaults to tkinter.filedialog)
            messagebox: tkinter messagebox module (optional, defaults to tkinter.messagebox)
        """
        # Use default tkinter modules if not provided
        if filedialog is None:
            from tkinter import filedialog
        if messagebox is None:
            from tkinter import messagebox

        # 1. Select Fusion Export File
        export_file_path = filedialog.askopenfilename(
            defaultextension=".fsexport",
            filetypes=[
                ("Fusion Export Files", "*.fsexport.txt;*.fsexport"),
                ("All Files", "*.*"),
            ],
            title="Select Fusion Export (.fsexport) File",
        )
        if not export_file_path:
            self.master_gui.status_label.config(
                text="Fusion export selection cancelled."
            )
            return

        markers = self._load_and_validate_fsexport(export_file_path)
        if markers is None:
            self.master_gui.status_label.config(text="Fusion export loading failed.")
            return

        # 2. Select Target Depth Map Folder
        target_folder = filedialog.askdirectory(
            title="Select Target Depth Map Folder"
        )
        if not target_folder:
            self.master_gui.status_label.config(
                text="Depth map folder selection cancelled."
            )
            return

        target_videos = self._scan_target_videos(target_folder)
        if target_videos is None or not target_videos:
            self.master_gui.status_label.config(text="No valid depth map videos found.")
            return

        # 3. Apply Parameters and Generate Sidecars
        sidecar_data, applied_count = self._apply_parameters(target_videos, markers)

        for file_data in target_videos:
            base_name_without_ext = os.path.splitext(file_data["full_path"])[0]
            json_filename = base_name_without_ext + ".fssidecar"

            if not self.sidecar_manager.save_sidecar_data(json_filename, sidecar_data):
                self.logger.error(
                    f"Failed to save sidecar for {file_data['basename']}."
                )

        # 4. Final Status
        if applied_count == 0:
            self.master_gui.status_label.config(
                text="Finished: No parameters were applied from the export file."
            )
        else:
            self.master_gui.status_label.config(
                text=f"Finished: Applied markers to {applied_count} files, "
                f"generated {len(target_videos)} FSSIDECARs."
            )
        messagebox.showinfo(
            "Sidecar Generation Complete",
            f"Successfully processed {os.path.basename(export_file_path)} "
            f"and generated {len(target_videos)} FSSIDECAR files.",
        )

    def generate_custom_sidecars(self, filedialog=None, messagebox=None) -> None:
        """Generate sidecars with custom names without requiring video files.

        Opens dialogs to select export file and output path, then generates
        indexed sidecar files for each marker.

        Args:
            filedialog: tkinter filedialog module (optional, defaults to tkinter.filedialog)
            messagebox: tkinter messagebox module (optional, defaults to tkinter.messagebox)
        """
        # Use default tkinter modules if not provided
        if filedialog is None:
            from tkinter import filedialog
        if messagebox is None:
            from tkinter import messagebox
        # 1. Select Fusion Export File
        export_file_path = filedialog.askopenfilename(
            defaultextension=".fsexport",
            filetypes=[
                ("Fusion Export Files", "*.fsexport.txt;*.fsexport"),
                ("All Files", "*.*"),
            ],
            title="Select Fusion Export (.fsexport) File",
        )
        if not export_file_path:
            self.master_gui.status_label.config(
                text="Fusion export selection cancelled."
            )
            return

        markers = self._load_and_validate_fsexport(export_file_path)
        if markers is None:
            self.master_gui.status_label.config(text="Fusion export loading failed.")
            return

        # 2. Select Output Sidecar Path
        custom_save_path = filedialog.asksaveasfilename(
            defaultextension=".fssidecar",
            filetypes=[("Sidecar Files", "*.fssidecar")],
            title="Save Sidecar As",
            initialfile=os.path.splitext(os.path.basename(export_file_path))[0],
        )
        if not custom_save_path:
            self.master_gui.status_label.config(text="Custom sidecar export cancelled.")
            return

        # 3. Process Markers
        applied_count = 0
        last_param_vals = {
            key: config["default"]
            for key, config in self.FUSION_PARAMETER_CONFIG.items()
        }

        for i, marker in enumerate(markers):
            current_param_vals = last_param_vals.copy()
            if marker.get("values"):
                marker_values = marker["values"]
                updated_from_marker = False
                for key, config in self.FUSION_PARAMETER_CONFIG.items():
                    fusion_key = config["fusion_key"]
                    if fusion_key in marker_values:
                        try:
                            current_param_vals[key] = config["type"](
                                marker_values[fusion_key]
                            )
                            updated_from_marker = True
                        except (ValueError, TypeError):
                            pass
                if updated_from_marker:
                    applied_count += 1

            # Prepare sidecar data
            sidecar_data = {}
            for key, config in self.FUSION_PARAMETER_CONFIG.items():
                value = current_param_vals[key]
                if config["type"] is bool:
                    sidecar_data[config["sidecar_key"]] = bool(value)
                elif config["type"] is str:
                    # String values (like border_mode) don't need rounding
                    sidecar_data[config["sidecar_key"]] = str(value)
                else:
                    sidecar_data[config["sidecar_key"]] = round(
                        float(value), config["decimals"]
                    )

            # Determine filename
            if len(markers) == 1:
                target_filename = custom_save_path
            else:
                base, ext = os.path.splitext(custom_save_path)
                target_filename = f"{base}_{i + 1:04d}{ext}"

            if not self.sidecar_manager.save_sidecar_data(
                target_filename, sidecar_data
            ):
                self.logger.error(f"Failed to save custom sidecar: {target_filename}")

            last_param_vals = current_param_vals.copy()

        # 4. Final Status
        self.master_gui.status_label.config(
            text=f"Finished: Generated {len(markers)} custom FSSIDECARs."
        )
        messagebox.showinfo(
            "Custom Export Complete",
            f"Successfully generated {len(markers)} custom FSSIDECAR files.",
        )
