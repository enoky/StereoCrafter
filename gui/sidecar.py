import os
import glob
import json
import math
import logging
from tkinter import filedialog, messagebox
from .utils import VideoFileClip

class FusionSidecarGenerator:
    """Handles parsing Fusion Export files, matching them to depth maps,
    and generating/saving FSSIDECAR files using carry-forward logic."""
    
    FUSION_PARAMETER_CONFIG = {
        # Key: {Label, Type, Default, FusionKey(fsexport), SidecarKey(fssidecar), Decimals}
        "convergence": {
            "label": "Convergence Plane", "type": float, "default": 0.5, 
            "fusion_key": "Convergence", "sidecar_key": "convergence_plane", "decimals": 3
        },
        "max_disparity": {
            "label": "Max Disparity", "type": float, "default": 35.0, 
            "fusion_key": "MaxDisparity", "sidecar_key": "max_disparity", "decimals": 1
        },
        "gamma": {
            "label": "Gamma Correction", "type": float, "default": 1.0,
            "fusion_key": "FrontGamma", "sidecar_key": "gamma", "decimals": 2
        },
        # These keys exist in the sidecar manager but are usually set in the source tool
        # We include them here for completeness if Fusion ever exported them
        "frame_overlap": {
            "label": "Frame Overlap", "type": float, "default": 3,
            "fusion_key": "Overlap", "sidecar_key": "frame_overlap", "decimals": 0
        },
        "input_bias": {
            "label": "Input Bias", "type": float, "default": 0.0, 
            "fusion_key": "Bias", "sidecar_key": "input_bias", "decimals": 2
        }
    }
    
    def __init__(self, master_gui, sidecar_manager):
        self.master_gui = master_gui
        self.sidecar_manager = sidecar_manager
        self.logger = logging.getLogger(__name__)

    def _get_video_frame_count(self, file_path):
        """Safely gets the frame count of a video file using moviepy."""
        try:
            clip = VideoFileClip(file_path)
            fps = clip.fps
            duration = clip.duration
            if fps is None or duration is None:
                # If moviepy failed to get reliable info, fall back
                fps = 24 
                if duration is None: return 0 
            
            frames = math.ceil(duration * fps)
            clip.close()
            return frames
        except Exception as e:
            self.logger.warning(f"Error getting frame count for {os.path.basename(file_path)}: {e}")
            return 0

    def _load_and_validate_fsexport(self, file_path):
        """Loads, parses, and validates marker data from a Fusion Export file."""
        try:
            with open(file_path, 'r') as f:
                export_data = json.load(f)
        except json.JSONDecodeError as e:
            messagebox.showerror("File Error", f"Failed to parse JSON in {os.path.basename(file_path)}: {e}")
            return None
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to read {os.path.basename(file_path)}: {e}")
            return None

        markers = export_data.get("markers", [])
        if not markers:
            messagebox.showwarning("Data Warning", "No 'markers' found in the export file.")
            return None
        
        # Sort markers by frame number (critical for carry-forward logic)
        markers.sort(key=lambda m: m['frame'])
        self.logger.info(f"Loaded {len(markers)} markers from {os.path.basename(file_path)}.")
        return markers

    def _scan_target_videos(self, folder):
        """Scans the target folder for video files and computes their frame counts."""
        video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
        found_files_paths = []
        for ext in video_extensions:
            found_files_paths.extend(glob.glob(os.path.join(folder, ext)))
        sorted_files_paths = sorted(found_files_paths)
        
        if not sorted_files_paths:
            messagebox.showwarning("No Files", f"No video depth map files found in: {folder}")
            return None

        target_video_data = []
        cumulative_frames = 0
        
        for full_path in sorted_files_paths:
            total_frames = self._get_video_frame_count(full_path)
            
            if total_frames == 0:
                self.logger.warning(f"Skipping {os.path.basename(full_path)} due to zero frame count.")
                continue

            target_video_data.append({
                "full_path": full_path,
                "basename": os.path.basename(full_path),
                "total_frames": total_frames,
                "timeline_start_frame": cumulative_frames,
                "timeline_end_frame": cumulative_frames + total_frames - 1,
            })
            cumulative_frames += total_frames
            
        self.logger.info(f"Scanned {len(target_video_data)} video files. Total timeline frames: {cumulative_frames}.")
        return target_video_data

    def generate_sidecars(self):
        """Main entry point for the Fusion Export to Sidecar generation workflow."""
        
        # 1. Select Fusion Export File
        export_file_path = filedialog.askopenfilename(
            defaultextension=".fsexport",
            filetypes=[("Fusion Export Files", "*.fsexport.txt;*.fsexport"), ("All Files", "*.*")],
            title="Select Fusion Export (.fsexport) File"
        )
        if not export_file_path:
            self.master_gui.status_label.config(text="Fusion export selection cancelled.")
            return

        markers = self._load_and_validate_fsexport(export_file_path)
        if markers is None:
            self.master_gui.status_label.config(text="Fusion export loading failed.")
            return

        # 2. Select Target Depth Map Folder
        target_folder = filedialog.askdirectory(title="Select Target Depth Map Folder")
        if not target_folder:
            self.master_gui.status_label.config(text="Depth map folder selection cancelled.")
            return

        target_videos = self._scan_target_videos(target_folder)
        if target_videos is None or not target_videos:
            self.master_gui.status_label.config(text="No valid depth map videos found.")
            return

        # 3. Apply Parameters (Carry-Forward Logic)
        applied_count = 0
        
        # Initialize last known values with the config defaults
        last_param_vals = {}
        for key, config in self.FUSION_PARAMETER_CONFIG.items():
             last_param_vals[key] = config["default"]

        for file_data in target_videos:
            file_start_frame = file_data["timeline_start_frame"]
            
            # Find the most relevant marker (latest marker frame <= file_start_frame)
            relevant_marker = None
            for marker in markers:
                if marker['frame'] <= file_start_frame:
                    relevant_marker = marker
                else:
                    break
            
            current_param_vals = last_param_vals.copy()

            if relevant_marker and relevant_marker.get('values'):
                marker_values = relevant_marker['values']
                updated_from_marker = False
                
                for key, config in self.FUSION_PARAMETER_CONFIG.items():
                    fusion_key = config["fusion_key"]
                    default_val = config["default"]
                    
                    if fusion_key in marker_values:
                        # Attempt to cast the value from the marker to the expected type
                        val = marker_values.get(fusion_key, default_val)
                        try:
                            current_param_vals[key] = config["type"](val)
                            updated_from_marker = True
                        except (ValueError, TypeError):
                            self.logger.warning(f"Marker value for '{fusion_key}' is invalid ({val}). Using previous/default value.")
                            
                if updated_from_marker:
                    applied_count += 1
            
            # 4. Save Sidecar JSON
            sidecar_data = {}
            for key, config in self.FUSION_PARAMETER_CONFIG.items():
                value = current_param_vals[key]
                # Round to configured decimals for clean sidecar output
                sidecar_data[config["sidecar_key"]] = round(value, config["decimals"])
                
            base_name_without_ext = os.path.splitext(file_data["full_path"])[0]
            json_filename = base_name_without_ext + ".fssidecar" # Target sidecar extension
            
            if not self.sidecar_manager.save_sidecar_data(json_filename, sidecar_data):
                self.logger.error(f"Failed to save sidecar for {file_data['basename']}.")

            # Update last values for carry-forward to the next file
            last_param_vals = current_param_vals.copy()

        # 5. Final Status
        if applied_count == 0:
            self.master_gui.status_label.config(text="Finished: No parameters were applied from the export file.")
        else:
            self.master_gui.status_label.config(text=f"Finished: Applied markers to {applied_count} files, generated {len(target_videos)} FSSIDECARs.")
        messagebox.showinfo("Sidecar Generation Complete", f"Successfully processed {os.path.basename(export_file_path)} and generated {len(target_videos)} FSSIDECAR files.")
