import threading
import gc
import os
import sys
import glob
import shutil
import json # Keep for config.json, help_content.json, settings.json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import queue
import time # Keep for time.perf_counter, time.strftime
import numpy as np
import torch
import message_catalog
# Import from the new message catalog
from message_catalog import (
    log_message as global_log_message, # Rename to avoid clash with class method if any
    set_gui_logger_callback,
    set_gui_verbosity,
    configure_timestamps as mc_configure_timestamps,
    INFO, DEBUG, WARNING, ERROR, CRITICAL, # Severity levels
    VERBOSITY_LEVEL_INFO, VERBOSITY_LEVEL_DEBUG, VERBOSITY_LEVEL_WARNING, # Verbosity levels
    VERBOSITY_LEVEL_SILENT
)
# Import the backend logic class
from depth_crafter_logic import DepthCrafterDemo

from depthcrafter.utils import (
    format_duration,
    # get_formatted_timestamp, # GUI will use message_catalog's or its own queue timestamping
    get_segment_output_folder_name,
    get_segment_npz_output_filename,
    get_full_video_output_filename,
    get_sidecar_json_filename,
    get_image_sequence_metadata, # New
    get_single_image_metadata,   # New
    define_video_segments, # This util now uses global_log_message
    load_json_file, # This util now uses global_log_message
    save_json_file, # This util now uses global_log_message
    save_depth_visual_as_mp4_util,
    save_depth_visual_as_png_sequence_util,
    save_depth_visual_as_exr_sequence_util,
    save_depth_visual_as_single_exr_util,
)

try:
    import merge_depth_segments
except ImportError as e:
    # Use global_log_message here if it's critical at module load time,
    # otherwise, a log during __init__ or an operation might be better.
    print(f"WARNING (gui): Could not import 'merge_depth_segments'. Merging functionality will not be available. Error: {e}")
    merge_depth_segments = None

try:
    import OpenEXR
    import Imath
    OPENEXR_AVAILABLE_GUI = True
except ImportError:
    OPENEXR_AVAILABLE_GUI = False
    # Log this using the catalog once it's set up, or print if critical before setup
    print("WARNING (gui): OpenEXR or Imath module not found. EXR options might be limited.")


from typing import Optional, Tuple, List, Dict

GUI_VERSION = "25.06.01"

class DepthCrafterGUI:
    CONFIG_FILENAME = "config.json"
    HELP_CONTENT_FILENAME = "help_content.json"
    MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION = True
    SETTINGS_FILETYPES = [("JSON files", "*.json"), ("All files", "*.*")]
    LAST_SETTINGS_DIR_CONFIG_KEY = "last_settings_dir"
    VIDEO_EXTENSIONS = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm", "*.flv", "*.gif"]
    IMAGE_EXTENSIONS = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.exr"]

    # Replaces the old self.log_message. This method's job is to put messages onto the GUI queue.
    def _queue_message_for_gui_log(self, message_string: str):
        """Queues a raw message string for display in the GUI log widget."""
        self.message_queue.put(("log_display", message_string))

    def log_message_if_verbose(self, message): # This was for resume logic
        # For resume logic, we can use DEBUG level messages.
        # The message_id should be specific to the resume action.
        # Example: global_log_message("GUI_RESUME_VERBOSE_CHECK", details=message)
        pass # Or log with a specific ID at DEBUG level

    def _get_segments_to_resume_or_overwrite(self, vid_path, original_basename, 
                                             segment_subfolder_path, all_potential_segments_from_define,
                                             base_job_info_for_video_ref: dict):
        master_meta_path = os.path.join(segment_subfolder_path, f"{original_basename}_master_meta.json")
        base_job_info_for_video_ref["pre_existing_successful_jobs"] = []

        if os.path.exists(master_meta_path):
            msg_dialog = (f"Master metadata found for '{original_basename}'. This video was previously processed/finalized.\n"
                          f"Path: {master_meta_path}\n\n"
                          f"Do you want to:\n"
                          f"- 'Yes': Re-process only FAILED segments and update master metadata?\n"
                          f"         (Existing successful segments will be preserved in the new master metadata).\n"
                          f"- 'No': Delete ALL existing segments and master metadata and start fresh?\n"
                          f"- 'Cancel': Skip this video entirely?")
            choice = messagebox.askyesnocancel("Resume or Overwrite Finalized Segments?", msg_dialog, parent=self.root)

            if choice is True:
                global_log_message("GUI_RESUME_REPROCESS_FAILED_MASTER_START", basename=original_basename)
                master_data = load_json_file(master_meta_path) # load_json_file uses global_log_message
                if not master_data or "jobs_info" not in master_data:
                    global_log_message("GUI_RESUME_MASTER_LOAD_FAIL_WARN", basename=original_basename)
                    choice = False # Fallthrough
                else:
                    failed_segment_jobs_to_run = []
                    successful_jobs_from_old_master = []
                    potential_segments_dict = {seg_job['segment_id']: seg_job for seg_job in all_potential_segments_from_define}

                    for job_in_meta in master_data.get("jobs_info", []):
                        seg_id = job_in_meta.get("segment_id")
                        if job_in_meta.get("status") == "success":
                            successful_jobs_from_old_master.append(job_in_meta)
                        elif seg_id is not None and seg_id in potential_segments_dict:
                            failed_segment_jobs_to_run.append(potential_segments_dict[seg_id])
                            global_log_message("GUI_RESUME_QUEUEING_FAILED_SEGMENT", segment_id=seg_id, basename=original_basename, status=job_in_meta.get('status', 'unknown'))
                        else:
                            global_log_message("GUI_RESUME_UNQUEUEABLE_SEGMENT_WARN", segment_id=seg_id, status=job_in_meta.get('status'), basename=original_basename)
                    
                    if not failed_segment_jobs_to_run:
                        global_log_message("GUI_RESUME_NO_FAILED_SEGMENTS_IN_MASTER", basename=original_basename)
                        base_job_info_for_video_ref["pre_existing_successful_jobs"] = successful_jobs_from_old_master
                        return [], "skipped_no_failed_segments_in_master_for_reprocessing"
                    
                    try:
                        backup_master_meta_path = master_meta_path + f".backup_{time.strftime('%Y%m%d%H%M%S')}"
                        shutil.move(master_meta_path, backup_master_meta_path)
                        global_log_message("GUI_BACKED_UP_FILE", original_path=os.path.basename(master_meta_path), backup_path=os.path.basename(backup_master_meta_path))
                    except Exception as e:
                        global_log_message("GUI_RESUME_BACKUP_MASTER_FAIL_WARN", error=str(e))

                    base_job_info_for_video_ref["pre_existing_successful_jobs"] = successful_jobs_from_old_master
                    return failed_segment_jobs_to_run, "reprocessing_failed_from_master"
            
            if choice is False: 
                global_log_message("GUI_RESUME_DELETING_FINALIZED_START", basename=original_basename, path=segment_subfolder_path)
                try:
                    if os.path.exists(segment_subfolder_path): shutil.rmtree(segment_subfolder_path)
                    global_log_message("GUI_RESUME_DELETING_FOLDER_SUCCESS", path=segment_subfolder_path)
                except Exception as e:
                    global_log_message("GUI_RESUME_DELETING_FOLDER_ERROR", path=segment_subfolder_path, error=str(e))
                return all_potential_segments_from_define, "overwriting_finalized"
            
            else: # Cancel
                global_log_message("GUI_RESUME_SKIPPING_FINALIZED", basename=original_basename)
                return [], "skipped_finalized"

        elif os.path.exists(segment_subfolder_path):
            msg_dialog_incomplete = (f"Incomplete segment data found for '{original_basename}' (no master metadata file).\n"
                                     f"Path: {segment_subfolder_path}\n\n"
                                     f"Do you want to:\n"
                                     f"- 'Yes': Resume by processing only missing/failed segments?\n"
                                     f"         (Existing successful segments will be preserved).\n"
                                     f"- 'No': Delete existing incomplete segments and start fresh?\n"
                                     f"- 'Cancel': Skip this video entirely?")
            choice_incomplete = messagebox.askyesnocancel("Resume Incomplete Segments?", msg_dialog_incomplete, parent=self.root)

            if choice_incomplete is True:
                global_log_message("GUI_RESUME_INCOMPLETE_START", basename=original_basename)
                segments_to_run = []
                num_already_complete = 0
                completed_segment_metadata_from_json = []

                for potential_segment_job in all_potential_segments_from_define:
                    seg_id = potential_segment_job["segment_id"]
                    total_segs = potential_segment_job["total_segments"]
                    expected_npz_filename = get_segment_npz_output_filename(original_basename, seg_id, total_segs)
                    expected_json_filename = get_sidecar_json_filename(expected_npz_filename)
                    npz_path = os.path.join(segment_subfolder_path, expected_npz_filename)
                    json_path = os.path.join(segment_subfolder_path, expected_json_filename)

                    is_complete_and_successful = False
                    if os.path.exists(npz_path) and os.path.exists(json_path):
                        segment_meta = load_json_file(json_path) # Uses global log, might be too verbose if log_message_if_verbose was intended for silence
                        if segment_meta and segment_meta.get("status") == "success":
                            is_complete_and_successful = True
                            num_already_complete += 1
                            completed_segment_metadata_from_json.append(segment_meta)
                        else:
                            status_msg = segment_meta.get('status', 'unknown') if segment_meta else 'JSON missing/corrupt'
                            global_log_message("GUI_RESUME_INCOMPLETE_SEGMENT_REPROCESS", segment_id=seg_id+1, total_segments=total_segs, basename=original_basename, status=status_msg)
                    else:
                        global_log_message("GUI_RESUME_INCOMPLETE_SEGMENT_MISSING", segment_id=seg_id+1, total_segments=total_segs, basename=original_basename, npz_filename=expected_npz_filename)

                    if not is_complete_and_successful:
                        segments_to_run.append(potential_segment_job)
                
                if num_already_complete > 0:
                    global_log_message("GUI_PRESERVING_SUCCESSFUL_SEGMENTS", num_complete=num_already_complete, video_name=original_basename)
                
                base_job_info_for_video_ref["pre_existing_successful_jobs"] = completed_segment_metadata_from_json

                if not segments_to_run and num_already_complete == len(all_potential_segments_from_define):
                    global_log_message("GUI_RESUME_ALL_SEGS_COMPLETE_NO_MASTER_WARN", basename=original_basename)
                    return [], "skipped_all_segments_found_complete_no_master"
                elif not segments_to_run and num_already_complete < len(all_potential_segments_from_define):
                     global_log_message("GUI_RESUME_NO_SEGS_TO_RUN_INCOMPLETE_WARN", basename=original_basename, total_defined=len(all_potential_segments_from_define), found_complete=num_already_complete)
                     return [], "skipped_no_segments_to_run_incomplete"
                return segments_to_run, "resuming_incomplete"

            elif choice_incomplete is False:
                global_log_message("GUI_RESUME_DELETING_INCOMPLETE_START", basename=original_basename, path=segment_subfolder_path)
                try:
                    if os.path.exists(segment_subfolder_path): shutil.rmtree(segment_subfolder_path)
                    global_log_message("GUI_RESUME_DELETING_FOLDER_SUCCESS", path=segment_subfolder_path)
                except Exception as e:
                    global_log_message("GUI_RESUME_DELETING_FOLDER_ERROR", path=segment_subfolder_path, error=str(e))
                return all_potential_segments_from_define, "overwriting_incomplete"
            
            else: # Cancel
                global_log_message("GUI_RESUME_SKIPPING_INCOMPLETE", basename=original_basename)
                return [], "skipped_incomplete"
                
        else: # Segment folder does not exist
            return all_potential_segments_from_define, "fresh_processing"

    def __init__(self, root):
        self.root = root
        self.root.title(f"DepthCrafter GUI Seg {GUI_VERSION}")
        self.input_dir_or_file_var = tk.StringVar(value=os.path.normpath("./input_clips"))
        self.output_dir = tk.StringVar(value=os.path.normpath("./output_depthmaps"))
        self.guidance_scale = tk.DoubleVar(value=1.0)
        self.inference_steps = tk.IntVar(value=5)
        self.max_res = tk.IntVar(value=960)
        self.seed = tk.IntVar(value=42)
        self.cpu_offload = tk.StringVar(value="model")
        self.use_cudnn_benchmark = tk.BooleanVar(value=True)
        self.process_length = tk.IntVar(value=-1)
        self.target_fps = tk.DoubleVar(value=-1.0)
        self.window_size = tk.IntVar(value=110)
        self.overlap = tk.IntVar(value=25)
        self.process_as_segments_var = tk.BooleanVar(value=False)
        self.save_final_output_json_var = tk.BooleanVar(value=False)
        self.merge_output_format_var = tk.StringVar(value="mp4")
        self.merge_alignment_method_var = tk.StringVar(value="Shift & Scale")
        self.merge_dither_var = tk.BooleanVar(value=False)
        self.merge_dither_strength_var = tk.DoubleVar(value=0.5)
        self.merge_gamma_correct_var = tk.BooleanVar(value=False)
        self.merge_gamma_value_var = tk.DoubleVar(value=1.5)
        self.merge_percentile_norm_var = tk.BooleanVar(value=False)
        self.merge_norm_low_perc_var = tk.DoubleVar(value=0.1)
        self.merge_norm_high_perc_var = tk.DoubleVar(value=99.9)
        self.keep_intermediate_npz_var = tk.BooleanVar(value=False)
        self.min_frames_to_keep_npz_var = tk.IntVar(value=0)
        self.keep_intermediate_segment_visual_format_var = tk.StringVar(value="mp4")
        self.merge_output_suffix_var = tk.StringVar(value="_depth") # New Variable
        self.merge_script_gui_silence_level_var = tk.StringVar(value="Normal (Info)") # This now maps to GUI verbosity
        self.current_input_mode = "batch_folder" # "batch_folder", "single_video_file", "single_image_file", "image_sequence_folder"
        self.single_file_mode_active = False # True if a single file/sequence folder is explicitly loaded
        self.effective_move_original_on_completion = self.MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION

        self.all_tk_vars = {
            "input_dir_or_file_var": self.input_dir_or_file_var, # Use new var name
            "output_dir": self.output_dir,
            "guidance_scale": self.guidance_scale,
            "inference_steps": self.inference_steps,
            "max_res": self.max_res,
            "seed": self.seed,
            "cpu_offload": self.cpu_offload,
            "use_cudnn_benchmark": self.use_cudnn_benchmark,
            "process_length": self.process_length,
            "target_fps": self.target_fps,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "process_as_segments_var": self.process_as_segments_var,
            "save_final_output_json_var": self.save_final_output_json_var,
            "merge_output_format_var": self.merge_output_format_var,
            "merge_alignment_method_var": self.merge_alignment_method_var,
            "merge_dither_var": self.merge_dither_var,
            "merge_dither_strength_var": self.merge_dither_strength_var,
            "merge_gamma_correct_var": self.merge_gamma_correct_var,
            "merge_gamma_value_var": self.merge_gamma_value_var,
            "merge_percentile_norm_var": self.merge_percentile_norm_var,
            "merge_norm_low_perc_var": self.merge_norm_low_perc_var,
            "merge_norm_high_perc_var": self.merge_norm_high_perc_var,
            "keep_intermediate_npz_var": self.keep_intermediate_npz_var,
            "min_frames_to_keep_npz_var": self.min_frames_to_keep_npz_var,
            "keep_intermediate_segment_visual_format_var": self.keep_intermediate_segment_visual_format_var,
            "merge_output_suffix_var": self.merge_output_suffix_var, # Add new var here
            "merge_script_gui_silence_level_var": self.merge_script_gui_silence_level_var, # Will be used to set GUI verbosity
        }
        self.initial_default_settings = self._collect_all_settings()
        self._help_data = None

        self.last_settings_dir = os.getcwd()
        self.message_queue = queue.Queue() # For GUI log updates

        # Set up message_catalog for GUI
        set_gui_logger_callback(self._queue_message_for_gui_log)
        set_gui_verbosity(self._get_mapped_gui_verbosity_level()) # Set initial GUI verbosity
        mc_configure_timestamps(console=True, gui=False) # Example: GUI adds its own timestamps via queue processor

        self.load_config() 
        self.stop_event = threading.Event()
        self.processing_thread = None
        self._create_menubar()
        self.create_widgets() 
        self.root.after(100, self.process_queue) # For GUI log updates
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.toggle_merge_related_options_active_state()

        # Initial log message using the new system, if desired
                
        global_log_message("GUI_INIT_COMPLETE") # THE ONE AND ONLY "INIT COMPLETE" LOG FROM __init__

    def _create_menubar(self):
        # ... (menubar creation unchanged) ...
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        self.file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=self.file_menu)
        self.file_menu.add_command(label="Load Settings...", command=self._load_all_settings)
        self.file_menu.add_command(label="Save Settings As...", command=self._save_all_settings_as)
        self.file_menu.add_command(label="Reset Settings to Default", command=self._reset_settings_to_defaults)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self.on_close)

        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="GUI Overview", command=lambda: self._show_help_for("general_gui_overview"))

    def _load_help_content(self):
        if self._help_data is None:
            self._help_data = load_json_file(DepthCrafterGUI.HELP_CONTENT_FILENAME) # Util handles logging
            if not self._help_data:
                self._help_data = {}
                global_log_message("GUI_HELP_LOAD_FAIL_WARN", filename=DepthCrafterGUI.HELP_CONTENT_FILENAME) # New ID
        return self._help_data

    def _show_help_for(self, help_key: str):
        # ... (unchanged, uses messagebox) ...
        help_content_store = self._load_help_content()
        content = help_content_store.get(help_key)
        if not content:
            messagebox.showinfo("Help Not Found", f"No help information available for '{help_key}'.\nEnsure '{DepthCrafterGUI.HELP_CONTENT_FILENAME}' is present and contains this key.")
            return

        help_title = content.get("title", "Help Information")
        help_text_str = content.get("text", "No details available.")
        help_window = tk.Toplevel(self.root)
        help_window.title(help_title)
        help_window.minsize(400, 200)
        help_window.transient(self.root)
        help_window.grab_set()
        text_frame = ttk.Frame(help_window, padding="10")
        text_frame.pack(expand=True, fill="both")
        help_text_widget = tk.Text(text_frame, wrap=tk.WORD, relief="flat", borderwidth=0, padx=5, pady=5, font=("Segoe UI", 9))
        help_text_widget.insert(tk.END, help_text_str)
        help_text_widget.config(state=tk.DISABLED)
        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=help_text_widget.yview)
        help_text_widget['yscrollcommand'] = scrollbar.set
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        help_text_widget.pack(side=tk.LEFT, expand=True, fill="both")
        button_frame = ttk.Frame(help_window, padding=(0, 5, 0, 10))
        button_frame.pack(fill=tk.X)
        ok_button = ttk.Button(button_frame, text="OK", command=help_window.destroy, width=10)
        ok_button.pack()
        self.root.update_idletasks()
        help_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - (help_window.winfo_width() // 2)
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - (help_window.winfo_height() // 2)
        help_window.geometry(f"+{x}+{y}")
        ok_button.focus_set()
        help_window.wait_window()

    def _collect_all_settings(self) -> dict:
        settings_data = {}
        for key, tk_var in self.all_tk_vars.items():
            try:
                value = tk_var.get()
                settings_data[key] = value
                if key == "target_fps": # Specific debug for target_fps
                    print(f"DEBUG: _collect_all_settings: target_fps raw value: {value}, type: {type(value)}")
            except tk.TclError:
                global_log_message("GUI_SETTINGS_GET_VAL_WARN", setting_key=key)
        return settings_data

    def _apply_all_settings(self, settings_data: dict):
        for key, value_from_json in settings_data.items():
            if key == "target_fps": # Specific debug
                print(f"DEBUG: _apply_all_settings: Loading target_fps from JSON. Value: {value_from_json}, Type: {type(value_from_json)}")
            if key in self.all_tk_vars:
                try:
                    self.all_tk_vars[key].set(value_from_json)
                    # After setting, get it back to see what DoubleVar stored
                    if key == "target_fps":
                        val_in_doublevar = self.all_tk_vars[key].get()
                        print(f"DEBUG: _apply_all_settings: target_fps in DoubleVar after set: {val_in_doublevar}, Type: {type(val_in_doublevar)}")
                except tk.TclError:
                     global_log_message("GUI_SETTINGS_SET_VAL_WARN", setting_key=key, value=value) # New ID
            else:
                global_log_message("GUI_SETTINGS_UNKNOWN_KEY_WARN", setting_key=key) # New ID
        if hasattr(self, 'process_as_segments_var'):
            self.toggle_merge_related_options_active_state()
        # Update GUI verbosity if it was part of loaded settings
        set_gui_verbosity(self._get_mapped_gui_verbosity_level())

    def _reset_settings_to_defaults(self):
        if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to their default values?"):
            self._apply_all_settings(self.initial_default_settings)
            global_log_message("GUI_SETTINGS_RESET") # Using existing ID

    def _load_all_settings(self):
        filepath = filedialog.askopenfilename(title="Load Settings File", filetypes=self.SETTINGS_FILETYPES, initialdir=self.last_settings_dir)
        if not filepath:
            global_log_message("GUI_ACTION_CANCELLED_BY_USER", action="Load settings") # New ID
            return
        self.last_settings_dir = os.path.dirname(filepath)
        settings_data = load_json_file(filepath) # Util handles logging
        if settings_data:
            self._apply_all_settings(settings_data)
            global_log_message("GUI_SETTINGS_LOADED", filepath=filepath) # Using existing ID
        else:
            messagebox.showerror("Load Error", f"Could not load settings from:\n{filepath}\nSee log for details.")

    def _save_all_settings_as(self):
        filepath = filedialog.asksaveasfilename(title="Save Settings As", filetypes=self.SETTINGS_FILETYPES, defaultextension=".json", initialdir=self.last_settings_dir)
        if not filepath:
            global_log_message("GUI_ACTION_CANCELLED_BY_USER", action="Save settings")
            return
        self.last_settings_dir = os.path.dirname(filepath)
        current_settings = self._collect_all_settings()
        if save_json_file(current_settings, filepath, indent=4): # Util handles logging
            global_log_message("GUI_SETTINGS_SAVED", filepath=filepath) # Using existing ID
            messagebox.showinfo("Save Successful", f"Settings saved to:\n{filepath}")
        else:
            messagebox.showerror("Save Error", f"Could not save settings to:\n{filepath}\nSee log for details.")
            # save_json_file already logged the specific error via global_log_message

    # ... (_add_param_with_help, _add_manual_help_icon unchanged) ...
    def _add_param_with_help(self, parent, label_text: str, var, row: int, help_key: str, entry_width=18, col_offset=0):
        tk.Label(parent, text=label_text).grid(row=row, column=0 + col_offset, sticky="e", padx=5, pady=2)
        entry = tk.Entry(parent, textvariable=var, width=entry_width)
        entry.grid(row=row, column=1 + col_offset, padx=(5,0), pady=2, sticky="w")
        help_label = tk.Label(parent, text="❓", fg="blue", cursor="hand2")
        help_label.grid(row=row, column=2 + col_offset, padx=(2,5), pady=2, sticky="w")
        help_label.bind("<Button-1>", lambda e, key=help_key: self._show_help_for(key))
        return entry, help_label

    def _add_manual_help_icon(self, parent, help_key: str, row: int, column: int, sticky="w", padx=(2,5), pady=2):
        help_label = tk.Label(parent, text="❓", fg="blue", cursor="hand2")
        help_label.bind("<Button-1>", lambda e, key=help_key: self._show_help_for(key))
        help_label.grid(row=row, column=column, padx=padx, pady=pady, sticky=sticky)
        return help_label

    def start_processing(self, video_processing_jobs, base_job_info_map):
        self.stop_event.clear()
        self.progress["value"] = 0
        self.progress["maximum"] = len(video_processing_jobs)
        global_log_message("GUI_PROCESSING_BATCH_START", num_jobs=len(video_processing_jobs)) # New ID

        try:
            if torch.cuda.is_available():
                 global_log_message("CUDA_AVAILABLE", device_name=torch.cuda.get_device_name(0))
            else:
                global_log_message("CUDA_UNAVAILABLE")
            demo = DepthCrafterDemo(
                unet_path="tencent/DepthCrafter",
                pre_train_path="stabilityai/stable-video-diffusion-img2vid-xt",
                cpu_offload=self.cpu_offload.get(),
                use_cudnn_benchmark=self.use_cudnn_benchmark.get(),
            )
        except Exception as e:
            global_log_message("MODEL_INIT_FAILURE", component="DepthCrafterDemo from GUI", reason=str(e))
            # self.message_queue.put(("log_display", "Processing cannot continue. Check model paths and dependencies."))
            # import traceback
            # self.message_queue.put(("log_display", traceback.format_exc()))
            # The above lines are now handled by global_log_message itself if it prints traceback
            return

        all_videos_master_metadata = {} 
        
        for i, job_info_to_run in enumerate(video_processing_jobs):
            if self.stop_event.is_set():
                global_log_message("GUI_CANCEL_REQUEST_HONORED") # New ID
                break
            
            current_video_path = job_info_to_run["video_path"] 
            original_basename = job_info_to_run["original_basename"]
            is_segment_job = job_info_to_run["is_segment"]

            if current_video_path not in all_videos_master_metadata:
                current_video_comprehensive_base_info = base_job_info_map.get(current_video_path, {})
                total_segments_for_this_video_overall = job_info_to_run.get("total_segments") if is_segment_job else 1
                if total_segments_for_this_video_overall is None and is_segment_job:
                    global_log_message("GUI_PROCESSING_MISSING_TOTAL_SEGS_WARN", basename=original_basename)
                    total_segments_for_this_video_overall = 1

                all_videos_master_metadata[current_video_path] = self._initialize_master_metadata_entry(
                    original_basename, 
                    job_info_to_run,
                    total_segments_for_this_video_overall
                )
                
                pre_existing_successful_segment_metadatas = current_video_comprehensive_base_info.get("pre_existing_successful_jobs", [])
                if pre_existing_successful_segment_metadatas:
                    global_log_message("GUI_PROCESSING_LOADING_PREEXISTING_SEGS", count=len(pre_existing_successful_segment_metadatas), basename=original_basename)
                    all_videos_master_metadata[current_video_path]["jobs_info"].extend(pre_existing_successful_segment_metadatas)
                    all_videos_master_metadata[current_video_path]["completed_successful_jobs"] += len(pre_existing_successful_segment_metadatas)
            
            master_meta_for_this_vid = all_videos_master_metadata[current_video_path]
            
            log_msg_prefix = f"Segment {job_info_to_run.get('segment_id', -1)+1}/{job_info_to_run.get('total_segments', 0)}" if is_segment_job else "Full video"
            # Using existing PROCESSING_JOB_PROGRESS
            global_log_message("PROCESSING_JOB_PROGRESS", item_name=original_basename, job_type=log_msg_prefix, current_job=i+1, total_jobs=len(video_processing_jobs))

            job_successful, returned_job_specific_metadata = self._process_single_job(demo, job_info_to_run, master_meta_for_this_vid)
            
            self.message_queue.put(("progress", i + 1)) 
            
            if is_segment_job and "segment_id" not in returned_job_specific_metadata:
                returned_job_specific_metadata["segment_id"] = job_info_to_run.get("segment_id", -1)
            
            if "_individual_metadata_path" in returned_job_specific_metadata:
                del returned_job_specific_metadata["_individual_metadata_path"]
            
            master_meta_for_this_vid["jobs_info"].append(returned_job_specific_metadata)
            
            if job_successful:
                master_meta_for_this_vid["completed_successful_jobs"] += 1
            else:
                master_meta_for_this_vid["completed_failed_jobs"] += 1
                
            total_accounted_for_vid = master_meta_for_this_vid["completed_successful_jobs"] + master_meta_for_this_vid["completed_failed_jobs"]
            
            if total_accounted_for_vid >= master_meta_for_this_vid["total_expected_jobs"]:
                self._finalize_video_processing(current_video_path, original_basename, master_meta_for_this_vid)

        if 'demo' in locals() and demo is not None:
            try:
                if hasattr(demo, 'pipe') and demo.pipe is not None:
                    if hasattr(demo.pipe, 'vae') and demo.pipe.vae is not None: del demo.pipe.vae
                    if hasattr(demo.pipe, 'unet') and demo.pipe.unet is not None: del demo.pipe.unet
                    del demo.pipe
                del demo
                global_log_message("MODEL_RELEASE_SUCCESS", component="DepthCrafter") # New ID
            except Exception as e_cleanup:
                global_log_message("MODEL_RELEASE_ERROR", component="DepthCrafter", error=str(e_cleanup)) # New ID
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            global_log_message("CUDA_CACHE_CLEARED")
        global_log_message("GUI_PROCESSING_BATCH_COMPLETE") # New ID

    def _initialize_master_metadata_entry(self, original_basename, job_info_for_original_details, total_expected_jobs_for_this_video):
        # ... (logic unchanged, just ensure no direct prints) ...
        entry = {
            "original_video_basename": original_basename,
            "original_video_details": {
                "raw_frame_count": job_info_for_original_details.get("original_video_raw_frame_count", 0),
                "original_fps": job_info_for_original_details.get("original_video_fps", 30.0)
            },
            "global_processing_settings": {
                "guidance_scale": self.guidance_scale.get(),
                "inference_steps": self.inference_steps.get(),
                "max_res_settings": self.max_res.get(),
                "seed_setting": self.seed.get(),
                "target_fps_setting": self.target_fps.get(),
                "process_max_frames_setting": self.process_length.get(),
                "gui_window_size_setting": self.window_size.get(),
                "gui_overlap_setting": self.overlap.get(),
                "processed_as_segments": self.process_as_segments_var.get(),
            },
            "jobs_info": [], "overall_status": "pending",
            "total_expected_jobs": total_expected_jobs_for_this_video,
            "completed_successful_jobs": 0, "completed_failed_jobs": 0,
        }
        if self.process_as_segments_var.get():
            entry["global_processing_settings"]["segment_definition_output_window_frames"] = job_info_for_original_details.get("gui_desired_output_window_frames")
            entry["global_processing_settings"]["segment_definition_output_overlap_frames"] = job_info_for_original_details.get("gui_desired_output_overlap_frames")
        return entry

    def _process_single_job(self, demo, job_info, master_meta_for_this_vid):
        # job_info is the complete dictionary for this specific job (segment or full)
        # It already contains "video_path", "source_type", "original_basename", "gui_fps_setting_at_definition", etc.
        # and if it's a segment, it has "segment_id", "num_frames_to_load_raw", etc.
        
        returned_job_specific_metadata = {}
        job_successful = False
        is_segment_job = job_info.get("is_segment", False) # Get from job_info
        original_basename = job_info["original_basename"]
        
        snapshotted_settings = master_meta_for_this_vid["global_processing_settings"]
        guidance_scale_for_job = snapshotted_settings["guidance_scale"]
        inference_steps_for_job = snapshotted_settings["inference_steps"]
        max_res_for_job = snapshotted_settings["max_res_settings"]
        seed_for_job = snapshotted_settings["seed_setting"]
        # target_fps_for_job is part of job_info["gui_fps_setting_at_definition"]

        # process_length_for_run: for full video, it's from GUI settings.
        # For segments, it's implicitly defined by job_info["num_frames_to_load_raw"].
        # demo.run's `process_length_for_read_full_video` is for the "full source" if not segmented.
        process_length_for_run_param = snapshotted_settings["process_max_frames_setting"] if not is_segment_job else -1
        
        window_size_for_pipe_call = snapshotted_settings["gui_window_size_setting"]
        overlap_for_pipe_call = snapshotted_settings["gui_overlap_setting"]

        try:
            keep_npz_for_this_job_run = False
            if is_segment_job:
                if self.keep_intermediate_npz_var.get():
                    min_frames_thresh = self.min_frames_to_keep_npz_var.get()
                    # Get orig_vid_frame_count from job_info itself, as it's part of base_job_info
                    orig_vid_frame_count = job_info.get("original_video_raw_frame_count", 0)
                    if min_frames_thresh <= 0 or orig_vid_frame_count >= min_frames_thresh:
                        keep_npz_for_this_job_run = True
            
            # Pass the *entire job_info* as the first argument to demo.run
            # If it's a segment, also pass it as segment_job_info_param for clarity in demo.run's logic.
            saved_data_filepath, returned_job_specific_metadata = demo.run(
                video_path_or_frames_or_info=job_info, # Pass the whole job spec here
                num_denoising_steps=inference_steps_for_job, 
                guidance_scale=guidance_scale_for_job,
                base_output_folder=self.output_dir.get(), 
                gui_window_size=window_size_for_pipe_call,
                gui_overlap=overlap_for_pipe_call, 
                process_length_for_read_full_video=process_length_for_run_param, 
                max_res=max_res_for_job, 
                seed=seed_for_job, 
                original_video_basename_override=original_basename, # Can also get from job_info
                segment_job_info_param=job_info if is_segment_job else None, # Pass job_info again if it's a segment
                keep_intermediate_npz_config=keep_npz_for_this_job_run,
                intermediate_segment_visual_format_config=self.keep_intermediate_segment_visual_format_var.get(),
                save_final_json_for_this_job_config=self.save_final_output_json_var.get()
            )
            if not returned_job_specific_metadata: # Should not happen if demo.run is robust
                returned_job_specific_metadata = {"status": "failure_no_metadata_from_run"}
                global_log_message("GUI_JOB_NO_METADATA_WARN", basename=original_basename) # New ID
            
            if saved_data_filepath and returned_job_specific_metadata.get("status") == "success":
                job_successful = True
            else: # Log failure here if not successful
                log_msg_prefix_local = f"Segment {job_info.get('segment_id', -1)+1}/{job_info.get('total_segments', 0)}" if is_segment_job else "Full video"
                global_log_message("GUI_JOB_STATUS_REPORT", basename=original_basename, job_prefix=log_msg_prefix_local, status=returned_job_specific_metadata.get('status', 'unknown_status')) # New ID
        
        except Exception as e:
            if not returned_job_specific_metadata: returned_job_specific_metadata = {}
            returned_job_specific_metadata["status"] = "exception_in_gui_process_single_job"
            returned_job_specific_metadata["error_message"] = str(e)
            log_msg_prefix_local = f"Segment {job_info.get('segment_id', -1)+1}/{job_info.get('total_segments', 0)}" if is_segment_job else "Full video"
            global_log_message("GUI_JOB_EXCEPTION", basename=original_basename, job_prefix=log_msg_prefix_local, error=str(e), traceback_info=sys.exc_info()) # New ID
        return job_successful, returned_job_specific_metadata

    def _determine_video_paths_and_processing_mode(self, original_basename, master_meta_for_this_vid):
        # ... (logic unchanged) ...
        main_output_dir_for_video = self.output_dir.get()
        was_processed_as_segments = master_meta_for_this_vid["global_processing_settings"]["processed_as_segments"]
        segment_subfolder_path = None
        if was_processed_as_segments:
            segment_subfolder_name = get_segment_output_folder_name(original_basename)
            segment_subfolder_path = os.path.join(main_output_dir_for_video, segment_subfolder_name)
        return main_output_dir_for_video, segment_subfolder_path, was_processed_as_segments

    def _finalize_video_processing(self, current_video_path, original_basename, master_meta_for_this_vid):
        # ... (status logic unchanged) ...
        if master_meta_for_this_vid["completed_failed_jobs"] == 0:
            master_meta_for_this_vid["overall_status"] = "all_success"
        elif master_meta_for_this_vid["completed_successful_jobs"] > 0:
            master_meta_for_this_vid["overall_status"] = "partial_success"
        else:
            master_meta_for_this_vid["overall_status"] = "all_failed"

        global_log_message("PROCESSING_VIDEO_FINAL_STATUS", basename=original_basename, status=master_meta_for_this_vid['overall_status']) # New ID
        
        main_output_dir, segment_subfolder_path, was_segments = self._determine_video_paths_and_processing_mode(original_basename, master_meta_for_this_vid)
        master_meta_filepath, merge_success, final_merged_path = None, False, "N/A (Merge not applicable or failed)"
        try:
            master_meta_filepath, meta_saved = self._save_master_metadata_and_cleanup_segment_json(master_meta_for_this_vid, original_basename, main_output_dir, was_segments, segment_subfolder_path)
            if was_segments and meta_saved and master_meta_for_this_vid["overall_status"] in ["all_success", "partial_success"]:
                merge_success, final_merged_path = self._handle_segment_merging(master_meta_filepath, original_basename, main_output_dir, master_meta_for_this_vid)
            elif was_segments:
                global_log_message("GUI_MERGE_SKIPPED_DUE_TO_STATUS", basename=original_basename, status=master_meta_for_this_vid['overall_status'], meta_saved=meta_saved, segment_folder=segment_subfolder_path or 'N/A') # New ID
            
            if self.save_final_output_json_var.get():
                self._save_final_output_sidecar_json(original_basename, final_merged_path, master_meta_filepath, master_meta_for_this_vid, was_segments, merge_success)
            
            if was_segments and segment_subfolder_path: # Check segment_subfolder_path existence
                self._cleanup_segment_folder(segment_subfolder_path, original_basename, master_meta_for_this_vid)
        except Exception as e:
            global_log_message("PROCESSING_FINALIZE_ERROR", item_name=original_basename, error_message=str(e)) # Using existing ID

        # Determine the final status to decide on moving the original source
        final_status = master_meta_for_this_vid.get("overall_status", "all_failed")

        # Decide if we should move the original file at all
        if self.effective_move_original_on_completion:
            # If moving is enabled (i.e., not single file mode), decide WHERE to move it
            
            target_subfolder_name = ""
            if final_status == "all_success":
                target_subfolder_name = "finished"
            elif final_status in ["partial_success", "all_failed"]:
                # If there were any failures at all, move to a "failed" folder
                target_subfolder_name = "failed"
            else:
                # Should not happen, but as a fallback, don't move it
                global_log_message("GUI_MOVE_UNKNOWN_STATUS_WARN", basename=original_basename, status=final_status)

            if target_subfolder_name:
                # Call the move function with the determined target subfolder
                self._move_original_source(current_video_path, original_basename, target_subfolder_name)
        else:
            # This is for single file mode, where we don't move the original
            global_log_message("GUI_ORIGINAL_VIDEO_MOVE_SKIPPED_SINGLE_MODE", basename=original_basename)

    def _move_original_source(self, current_video_path: str, original_basename: str, target_subfolder: str):
        # target_subfolder will be "finished" or "failed"
        global_log_message("GUI_ORIGINAL_SOURCE_MOVE_ATTEMPT", basename=original_basename, target_folder=target_subfolder)
        try:
            # --- This logic is from your last working version of the move function ---
            path_from_gui_input_field = self.input_dir_or_file_var.get()

            actual_input_root_for_target_folder: str
            if os.path.isdir(path_from_gui_input_field):
                actual_input_root_for_target_folder = path_from_gui_input_field
            elif os.path.isfile(path_from_gui_input_field):
                actual_input_root_for_target_folder = os.path.dirname(path_from_gui_input_field)
            else:
                log_message("GUI_ORIGINAL_SOURCE_MOVE_INVALID_INPUT_ROOT_WARN", gui_input_path=path_from_gui_input_field)
                actual_input_root_for_target_folder = os.path.dirname(current_video_path)
                if not os.path.isdir(actual_input_root_for_target_folder):
                    log_message("GUI_ORIGINAL_SOURCE_MOVE_CANNOT_DETERMINE_ROOT", path=current_video_path)
                    global_log_message("GUI_ORIGINAL_SOURCE_MOVE_ERROR", basename=original_basename, error="Cannot determine valid root for target folder.", traceback_info=None)
                    return
            # --- End of logic from last working version ---

            # Use the target_subfolder parameter to create the destination directory
            destination_dir = os.path.join(actual_input_root_for_target_folder, target_subfolder)
            os.makedirs(destination_dir, exist_ok=True)
            
            dest_filename = os.path.basename(current_video_path)
            dest_path = os.path.join(destination_dir, dest_filename)

            if os.path.exists(current_video_path):
                if os.path.exists(dest_path):
                    base, ext = os.path.splitext(dest_filename) 
                    new_dest_name = f"{base}{time.strftime('_%Y%m%d%H%M%S')}{ext}"
                    dest_path = os.path.join(destination_dir, new_dest_name)
                    global_log_message("GUI_ORIGINAL_SOURCE_MOVE_RENAMED", old_name=dest_filename, new_name=new_dest_name)
                
                shutil.move(current_video_path, dest_path)
                global_log_message("GUI_ORIGINAL_SOURCE_MOVE_SUCCESS", filename=dest_filename, destination_folder=target_subfolder)
            else:
                global_log_message("GUI_ORIGINAL_SOURCE_MOVE_SOURCE_NOT_FOUND", path=current_video_path)
        except Exception as e:
            global_log_message("GUI_ORIGINAL_SOURCE_MOVE_ERROR", basename=original_basename, error=str(e), traceback_info=sys.exc_info())

    def _save_master_metadata_and_cleanup_segment_json(self, master_meta_to_save, original_basename, main_output_dir, was_segments, segment_subfolder_path):
        master_meta_filepath, meta_saved = None, False
        if was_segments:
            if not segment_subfolder_path:
                segment_subfolder_path = os.path.join(main_output_dir, get_segment_output_folder_name(original_basename))
            os.makedirs(segment_subfolder_path, exist_ok=True)
            master_meta_filepath = os.path.join(segment_subfolder_path, f"{original_basename}_master_meta.json")
        else:
            master_meta_filepath = os.path.join(main_output_dir, f"{original_basename}_master_meta.json")

        should_save_master_meta_here = was_segments
        if should_save_master_meta_here:
            if save_json_file(master_meta_to_save, master_meta_filepath): # Util logs success/fail
                global_log_message("GUI_MASTER_META_SAVED", basename=original_basename, path=master_meta_filepath) # New ID
                meta_saved = True
            if was_segments and meta_saved and segment_subfolder_path:
                global_log_message("GUI_CLEANUP_INDIVIDUAL_SEG_JSON_START", basename=original_basename) # New ID
                deleted_count = 0
                for job_data in master_meta_to_save.get("jobs_info", []):
                    npz_file = job_data.get("output_segment_filename")
                    if npz_file:
                        json_to_del = os.path.join(segment_subfolder_path, get_sidecar_json_filename(npz_file))
                        if os.path.exists(json_to_del):
                            try: os.remove(json_to_del); deleted_count += 1
                            except Exception as e: global_log_message("GUI_CLEANUP_INDIVIDUAL_SEG_JSON_ERROR", path=json_to_del, error=str(e)) # New ID
                global_log_message("GUI_CLEANUP_INDIVIDUAL_SEG_JSON_SUMMARY", count=deleted_count) # New ID
            elif was_segments and not meta_saved:
                global_log_message("GUI_CLEANUP_INDIVIDUAL_SEG_JSON_SKIPPED_NO_MASTER", basename=original_basename) # New ID
        elif not was_segments:
            global_log_message("GUI_MASTER_META_SAVE_SKIPPED_FULL_VIDEO", basename=original_basename, path=os.path.basename(master_meta_filepath)) # New ID
            meta_saved = False
        return master_meta_filepath, meta_saved

    def _handle_segment_merging(self, master_meta_filepath, original_basename, main_output_dir, master_meta):
        if not merge_depth_segments:
            global_log_message("MERGE_MODULE_UNAVAILABLE", video_name=original_basename) # Use existing
            return False, "N/A (Merge module not available)"
        
        out_fmt = self.merge_output_format_var.get()
        output_suffix = self.merge_output_suffix_var.get() # Get the suffix
        merged_base_name = f"{original_basename}{output_suffix}" # Use suffix

        align_method = "linear_blend" if self.merge_alignment_method_var.get() == "Linear Blend" else "shift_scale"
        merged_ok, actual_path = False, None
        try:
            actual_path = merge_depth_segments.merge_depth_segments(
                master_meta_path=master_meta_filepath, output_path_arg=main_output_dir,
                do_dithering=self.merge_dither_var.get(), dither_strength_factor=self.merge_dither_strength_var.get(),
                apply_gamma_correction=self.merge_gamma_correct_var.get(), gamma_value=self.merge_gamma_value_var.get(),
                use_percentile_norm=self.merge_percentile_norm_var.get(), norm_low_percentile=self.merge_norm_low_perc_var.get(),
                norm_high_percentile=self.merge_norm_high_perc_var.get(), output_format=out_fmt,
                merge_alignment_method=align_method, 
                output_filename_override_base=merged_base_name 
            )
            merged_ok = bool(actual_path)
        except Exception as e: 
            global_log_message("GUI_MERGE_CALL_EXCEPTION", basename=original_basename, error=str(e), traceback_info=sys.exc_info()) 
        
        final_path_to_return = "N/A (Merge failed/path not returned)"
        if merged_ok and actual_path:
            final_path_to_return = actual_path
        elif merged_ok : 
             global_log_message("GUI_MERGE_PATH_RETURN_ISSUE_WARN", basename=original_basename) 
             # Construct expected filename if actual_path is None but merged_ok is True
             effective_out_fmt = out_fmt.replace("_main10", "")
             final_path_to_return = os.path.join(main_output_dir, f"{merged_base_name}.{effective_out_fmt}") 
        
        return merged_ok, final_path_to_return

    def _save_final_output_sidecar_json(self, original_basename, final_merged_path, master_meta_filepath, master_meta, was_segments, merge_successful):
        json_path, json_content = None, {}
        output_suffix_val = self.merge_output_suffix_var.get() # Get suffix for consistency if needed

        if was_segments:
            if merge_successful and final_merged_path and not final_merged_path.startswith("N/A"):
                out_fmt_selected = self.merge_output_format_var.get()
                
                # The final_output_path is the actual path, extension is part of it.
                # The final_output_format should reflect the selection.
                json_content = {
                    "source_video_basename": original_basename, "processing_mode": "segmented_then_merged",
                    "final_output_path": os.path.abspath(final_merged_path), 
                    "final_output_format_selected": out_fmt_selected,
                    "master_metadata_path_source": os.path.abspath(master_meta_filepath) if master_meta_filepath else None,
                    "global_processing_settings_summary": master_meta.get("global_processing_settings"),
                    "merge_settings_summary": {
                        "output_format_selected": out_fmt_selected, 
                        "output_suffix": output_suffix_val,
                        "alignment_method": self.merge_alignment_method_var.get(),
                        "dithering": self.merge_dither_var.get(), "dither_strength": self.merge_dither_strength_var.get(),
                        "gamma_correction": self.merge_gamma_correct_var.get(),
                        "gamma_value_if_applied": self.merge_gamma_value_var.get() if self.merge_gamma_correct_var.get() else 1.0,
                        "percentile_norm": self.merge_percentile_norm_var.get(),
                        "norm_low_perc": self.merge_norm_low_perc_var.get(), "norm_high_perc": self.merge_norm_high_perc_var.get(),
                    }, "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                }
                if os.path.isdir(final_merged_path):
                    json_path = os.path.join(os.path.dirname(final_merged_path.rstrip(os.sep)), f"{os.path.basename(final_merged_path.rstrip(os.sep))}.json")
                elif os.path.isfile(final_merged_path):
                    json_path = get_sidecar_json_filename(final_merged_path)
                else: global_log_message("GUI_FINAL_JSON_PATH_DETERMINE_ERROR_MERGED", basename=original_basename, path=final_merged_path) 
            else: global_log_message("GUI_FINAL_JSON_SKIPPED_MERGE_FAIL", basename=original_basename) 
        else: # Full video (output_suffix doesn't apply here, _depth is hardcoded in get_full_video_output_filename)
            if master_meta and master_meta.get("jobs_info"):
                job_info = master_meta["jobs_info"][0]
                relative_output_filename = job_info.get("output_video_filename") 
                if relative_output_filename:
                    out_path = os.path.join(self.output_dir.get(), relative_output_filename)
                    out_fmt_from_ext = os.path.splitext(relative_output_filename)[1].lstrip('.') 

                    json_content = {
                        "source_video_basename": original_basename, "processing_mode": "full_video",
                        "final_output_path": os.path.abspath(out_path), 
                        "final_output_format": out_fmt_from_ext, # Actual format from extension
                        "global_processing_settings": master_meta.get("global_processing_settings"),
                        "job_specific_details": job_info,
                        "generation_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                    }
                    if os.path.isdir(out_path): 
                         json_path = os.path.join(os.path.dirname(out_path.rstrip(os.sep)), f"{os.path.basename(out_path.rstrip(os.sep))}.json")
                    elif os.path.isfile(out_path):
                        json_path = get_sidecar_json_filename(out_path)
                    else: global_log_message("GUI_FINAL_JSON_PATH_DETERMINE_ERROR_FULL", basename=original_basename, path=out_path) 
                else: global_log_message("GUI_FINAL_JSON_SKIPPED_FULL_NO_PATH_INFO", basename=original_basename) 
            else: global_log_message("GUI_FINAL_JSON_SKIPPED_FULL_NO_MASTER_META", basename=original_basename) 

        if json_path and json_content:
            global_log_message("GUI_FINAL_JSON_SAVE_ATTEMPT", path=json_path) 
            if save_json_file(json_content, json_path): 
                global_log_message("GUI_FINAL_JSON_SAVE_SUCCESS", path=json_path) 
        elif self.save_final_output_json_var.get():
            mode = "merged" if was_segments else "full video"
            global_log_message("GUI_FINAL_JSON_NOT_CREATED_WARN", mode=mode, basename=original_basename)

    def _cleanup_segment_folder(self, segment_subfolder_path, original_basename, master_meta):
        del_folder = False
        if not self.keep_intermediate_npz_var.get():
            global_log_message("GUI_CLEANUP_SEG_FOLDER_NO_KEEP_NPZ", basename=original_basename) # New ID
            del_folder = True
        else:
            min_frames = self.min_frames_to_keep_npz_var.get()
            if min_frames > 0:
                orig_frames = master_meta.get("original_video_details", {}).get("raw_frame_count", 0)
                if orig_frames < min_frames:
                    global_log_message("GUI_CLEANUP_SEG_FOLDER_THRESHOLD_NOT_MET", basename=original_basename, frames=orig_frames, threshold=min_frames) # New ID
                    del_folder = True
                else:
                    global_log_message("GUI_CLEANUP_SEG_FOLDER_THRESHOLD_MET_KEPT", basename=original_basename, frames=orig_frames, threshold=min_frames) # New ID
            else:
                global_log_message("GUI_CLEANUP_SEG_FOLDER_KEPT_NO_THRESHOLD", basename=original_basename) # New ID
        if del_folder:
            if os.path.exists(segment_subfolder_path):
                try: 
                    shutil.rmtree(segment_subfolder_path)
                    global_log_message("GUI_CLEANUP_SEG_FOLDER_DELETE_SUCCESS", basename=original_basename) # New ID
                except Exception as e:
                    global_log_message("GUI_CLEANUP_SEG_FOLDER_DELETE_ERROR", path=segment_subfolder_path, error=str(e)) # New ID
            else:
                global_log_message("GUI_CLEANUP_SEG_FOLDER_NOT_FOUND_WARN", path=segment_subfolder_path) # New ID
        else:
            global_log_message("GUI_CLEANUP_SEG_FOLDER_KEPT_INFO", path=segment_subfolder_path) # New ID

    def create_widgets(self):
        # ... (widget creation mostly unchanged, ensure no direct log calls) ...
        # Call to update GUI verbosity after combo box for it is created, if it exists
        if hasattr(self, 'combo_merge_verbosity'): # Assuming this is the Tk var for merge_script_gui_silence_level_var
            self.combo_merge_verbosity.bind("<<ComboboxSelected>>", self._update_gui_verbosity_from_combobox)
        # ... (rest of create_widgets)
        if not hasattr(self, 'widgets_to_disable_during_processing'):
            self.widgets_to_disable_during_processing = []
        gui_verbosity_options = [
            "Verbose (Detail)", 
            "Normal (Info)", 
            "Less Verbose (Warnings)", 
            "Silent (Errors Only)"
            # "Developer (Debug)" # Optional
        ]

        dir_frame = tk.LabelFrame(self.root, text="Input Source") # Renamed frame
        dir_frame.pack(fill="x", padx=10, pady=5, expand=False)        
        tk.Label(dir_frame, text="Input Folder/File:").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        self.entry_input_dir_or_file = tk.Entry(dir_frame, textvariable=self.input_dir_or_file_var, width=50) # Use new var
        self.entry_input_dir_or_file.grid(row=0, column=1, padx=5, pady=2, sticky="ew")        
        browse_buttons_frame = tk.Frame(dir_frame)
        browse_buttons_frame.grid(row=0, column=2, padx=5, pady=0, sticky="w")
        self.browse_input_folder_btn = tk.Button(browse_buttons_frame, text="Browse Folder", command=self.browse_input_folder)
        self.browse_input_folder_btn.pack(side=tk.LEFT, padx=(0,2))        
        self.browse_single_file_btn = tk.Button(browse_buttons_frame, text="Load Single File", command=self.browse_single_input_file) # New button
        self.browse_single_file_btn.pack(side=tk.LEFT, padx=(2,0))
        dir_frame.columnconfigure(1, weight=1) # Allow entry to expand
        self.widgets_to_disable_during_processing.extend([self.entry_input_dir_or_file, self.browse_input_folder_btn, self.browse_single_file_btn])

        tk.Label(dir_frame, text="Output Folder:").grid(row=1, column=0, sticky="e", padx=5, pady=2)
        self.entry_output_dir = tk.Entry(dir_frame, textvariable=self.output_dir, width=50)
        self.entry_output_dir.grid(row=1, column=1, padx=5, pady=2)
        self.browse_output_btn = tk.Button(dir_frame, text="Browse", command=self.browse_output)
        self.browse_output_btn.grid(row=1, column=2, padx=5, pady=2)
        self.widgets_to_disable_during_processing.extend([self.entry_output_dir, self.browse_output_btn])

        top_controls_outer_frame = tk.Frame(self.root)
        top_controls_outer_frame.pack(fill="x", padx=0, pady=0, expand=False) 

        main_params_frame = tk.LabelFrame(top_controls_outer_frame, text="Main Parameters")
        main_params_frame.grid(row=0, column=0, padx=(10,5), pady=5, sticky="nw")
        row_idx = 0
        e, hl = self._add_param_with_help(main_params_frame, "Guidance Scale:", self.guidance_scale, row_idx, "guidance_scale"); self.widgets_to_disable_during_processing.extend([e, hl]); row_idx += 1
        e, hl = self._add_param_with_help(main_params_frame, "Inference Steps:", self.inference_steps, row_idx, "inference_steps"); self.widgets_to_disable_during_processing.extend([e, hl]); row_idx += 1
        e, hl = self._add_param_with_help(main_params_frame, "Max Resolution:", self.max_res, row_idx, "max_res"); self.widgets_to_disable_during_processing.extend([e, hl]); row_idx += 1
        e, hl = self._add_param_with_help(main_params_frame, "Seed:", self.seed, row_idx, "seed"); self.widgets_to_disable_during_processing.extend([e, hl]); row_idx += 1

        tk.Label(main_params_frame, text="CPU Offload Mode:").grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.combo_cpu_offload = ttk.Combobox(main_params_frame, textvariable=self.cpu_offload, values=["model", "sequential", ""], width=17, state="readonly")
        self.combo_cpu_offload.grid(row=row_idx, column=1, padx=5, pady=2, sticky="w")
        hl_cpu = self._add_manual_help_icon(main_params_frame, "cpu_offload", row_idx, 2)
        self.widgets_to_disable_during_processing.extend([self.combo_cpu_offload, hl_cpu]); row_idx += 1

        self.cudnn_benchmark_cb = tk.Checkbutton(main_params_frame, text="Enable cuDNN Benchmark (Nvidia GPU Only)", variable=self.use_cudnn_benchmark)
        self.cudnn_benchmark_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=(5, 0), pady=2)
        hl_cudnn = self._add_manual_help_icon(main_params_frame, "cudnn_benchmark", row_idx, 2, sticky="w", padx=(0,5))
        self.widgets_to_disable_during_processing.extend([self.cudnn_benchmark_cb, hl_cudnn]); row_idx +=1

        self.save_final_json_cb = tk.Checkbutton(main_params_frame, text="Save Sidecar JSON for Final Output", variable=self.save_final_output_json_var)
        self.save_final_json_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        hl_sfj = self._add_manual_help_icon(main_params_frame, "save_final_json", row_idx, 2, sticky="w", padx=(0,5))
        self.widgets_to_disable_during_processing.extend([self.save_final_json_cb, hl_sfj]); row_idx +=1

        fs_frame = tk.LabelFrame(top_controls_outer_frame, text="Frame & Segment Control")
        fs_frame.grid(row=0, column=1, padx=(5,10), pady=5, sticky="new")
        top_controls_outer_frame.columnconfigure(0, weight=1)
        top_controls_outer_frame.columnconfigure(1, weight=1)

        row_idx = 0 
        e, hl = self._add_param_with_help(fs_frame, "Window Size:", self.window_size, row_idx, "window_size"); self.widgets_to_disable_during_processing.extend([e, hl]); row_idx += 1
        e, hl = self._add_param_with_help(fs_frame, "Overlap:", self.overlap, row_idx, "overlap"); self.widgets_to_disable_during_processing.extend([e, hl]); row_idx += 1
        e, hl = self._add_param_with_help(fs_frame, "Target FPS (-1 Original):", self.target_fps, row_idx, "target_fps"); self.widgets_to_disable_during_processing.extend([e, hl]); row_idx += 1
        e, hl = self._add_param_with_help(fs_frame, "Process Max Frames (-1 All):", self.process_length, row_idx, "process_length"); self.widgets_to_disable_during_processing.extend([e, hl]); row_idx += 1
        self.process_as_segments_cb = tk.Checkbutton(fs_frame, text="Process as Segments (Low VRAM Mode)", variable=self.process_as_segments_var, command=self.toggle_merge_related_options_active_state)
        self.process_as_segments_cb.grid(row=row_idx, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        hl_pas = self._add_manual_help_icon(fs_frame, "process_as_segments", row_idx, 2, sticky="w", padx=(0,5))
        self.widgets_to_disable_during_processing.extend([self.process_as_segments_cb, hl_pas]); row_idx += 1

        merge_opts_frame = tk.LabelFrame(self.root, text="Merged Output Options (if segments processed)")
        merge_opts_frame.pack(fill="x", padx=10, pady=5, expand=False)
        merge_opts_frame.columnconfigure(0, minsize=240)
        self.merge_related_widgets_references = []
        self.keep_npz_dependent_widgets = []
        row_idx = 0

        # Keep intermediate NPZ options GUI
        self.keep_npz_cb = tk.Checkbutton(merge_opts_frame, text="Keep intermediate NPZ files", variable=self.keep_intermediate_npz_var, command=self.toggle_keep_npz_dependent_options_state)
        self.keep_npz_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        hl_knpz = self._add_manual_help_icon(merge_opts_frame, "keep_npz", row_idx, 2, sticky="w", padx=(0,5))
        self.merge_related_widgets_references.append((self.keep_npz_cb, hl_knpz))
        self.widgets_to_disable_during_processing.extend([self.keep_npz_cb, hl_knpz]); row_idx += 1

        self.lbl_min_frames_npz = tk.Label(merge_opts_frame, text="  ↳ Min Orig. Vid Frames to Keep NPZ:")
        self.lbl_min_frames_npz.grid(row=row_idx, column=0, sticky="e", padx=(20,2), pady=2)
        self.entry_min_frames_npz = tk.Entry(merge_opts_frame, textvariable=self.min_frames_to_keep_npz_var, width=7)
        self.entry_min_frames_npz.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        hl_mfn = self._add_manual_help_icon(merge_opts_frame, "min_frames_npz", row_idx, 2, sticky="w", padx=(0,5))
        self.keep_npz_dependent_widgets.extend([self.lbl_min_frames_npz, self.entry_min_frames_npz, hl_mfn])
        self.widgets_to_disable_during_processing.extend([self.lbl_min_frames_npz, self.entry_min_frames_npz, hl_mfn]); row_idx += 1

        # Segment Visual Format options GUI
        self.lbl_intermediate_fmt = tk.Label(merge_opts_frame, text="  ↳ Segment Visual Format:")
        self.lbl_intermediate_fmt.grid(row=row_idx, column=0, sticky="e", padx=(20,2), pady=2)
        combo_intermediate_fmt_values = ["png_sequence", "mp4", "main10_mp4", "none"]
        if OPENEXR_AVAILABLE_GUI: combo_intermediate_fmt_values.extend(["exr_sequence", "exr"])
        self.combo_intermediate_fmt = ttk.Combobox(merge_opts_frame, textvariable=self.keep_intermediate_segment_visual_format_var, values=combo_intermediate_fmt_values, width=17, state="readonly")
        self.combo_intermediate_fmt.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        hl_sif = self._add_manual_help_icon(merge_opts_frame, "segment_visual_format", row_idx, 2, sticky="w", padx=(0,5))
        self.keep_npz_dependent_widgets.extend([self.lbl_intermediate_fmt, self.combo_intermediate_fmt, hl_sif])
        self.widgets_to_disable_during_processing.extend([self.lbl_intermediate_fmt, self.combo_intermediate_fmt, hl_sif]); row_idx += 1
        self.toggle_keep_npz_dependent_options_state() # Call after creation

        # Dithering, Gamma, Percentile Norm options remain the same, ensure row_idx continues correctly
        self.merge_dither_cb = tk.Checkbutton(merge_opts_frame, text="Dithering (MP4)", variable=self.merge_dither_var, command=self.toggle_dither_options_active_state)
        self.merge_dither_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        dither_details_frame = tk.Frame(merge_opts_frame)
        dither_details_frame.grid(row=row_idx, column=1, sticky="w", padx=(0,0)) # Ensure it doesn't span too far
        self.lbl_dither_str = tk.Label(dither_details_frame, text="Strength:")
        self.lbl_dither_str.pack(side=tk.LEFT, padx=(0, 2))
        self.entry_dither_str = tk.Entry(dither_details_frame, textvariable=self.merge_dither_strength_var, width=7)
        self.entry_dither_str.pack(side=tk.LEFT, padx=(0, 0))
        hl_mds = self._add_manual_help_icon(merge_opts_frame, "merge_dither", row_idx, 2, sticky="w", padx=(2,5))
        self.merge_related_widgets_references.append((self.merge_dither_cb, dither_details_frame, hl_mds))
        self.widgets_to_disable_during_processing.extend([self.merge_dither_cb, self.lbl_dither_str, self.entry_dither_str, hl_mds]); row_idx += 1

        # Gamma options GUI
        self.merge_gamma_cb = tk.Checkbutton(merge_opts_frame, text="Gamma Correct (MP4)", variable=self.merge_gamma_correct_var, command=self.toggle_gamma_options_active_state)
        self.merge_gamma_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        gamma_details_frame = tk.Frame(merge_opts_frame)
        gamma_details_frame.grid(row=row_idx, column=1, sticky="w", padx=(0,0))
        self.lbl_gamma_val = tk.Label(gamma_details_frame, text="Value:")
        self.lbl_gamma_val.pack(side=tk.LEFT, padx=(0, 2))
        self.entry_gamma_val = tk.Entry(gamma_details_frame, textvariable=self.merge_gamma_value_var, width=7)
        self.entry_gamma_val.pack(side=tk.LEFT, padx=(0, 0))
        hl_mgv = self._add_manual_help_icon(merge_opts_frame, "merge_gamma", row_idx, 2, sticky="w", padx=(2,5))
        self.merge_related_widgets_references.append((self.merge_gamma_cb, gamma_details_frame, hl_mgv))
        self.widgets_to_disable_during_processing.extend([self.merge_gamma_cb, self.lbl_gamma_val, self.entry_gamma_val, hl_mgv]); row_idx += 1

        # Percentile Normalization options GUI
        self.merge_perc_norm_cb = tk.Checkbutton(merge_opts_frame, text="Percentile Normalization", variable=self.merge_percentile_norm_var, command=self.toggle_percentile_norm_options_active_state)
        self.merge_perc_norm_cb.grid(row=row_idx, column=0, sticky="w", padx=5, pady=2)
        hl_mpncb = self._add_manual_help_icon(merge_opts_frame, "merge_percentile_norm", row_idx, 2, sticky="w", padx=(0,5))
        self.merge_related_widgets_references.append((self.merge_perc_norm_cb, hl_mpncb))
        self.widgets_to_disable_during_processing.extend([self.merge_perc_norm_cb, hl_mpncb]); row_idx += 1
        
        low_high_frame = tk.Frame(merge_opts_frame)
        low_high_frame.grid(row=row_idx, column=1, sticky="w", padx=(0,0))
        self.lbl_low_perc = tk.Label(low_high_frame, text="Low:")
        self.lbl_low_perc.pack(side=tk.LEFT, padx=(0,2))
        self.entry_low_perc = tk.Entry(low_high_frame, textvariable=self.merge_norm_low_perc_var, width=7)
        self.entry_low_perc.pack(side=tk.LEFT, padx=(0,10))
        self.lbl_high_perc = tk.Label(low_high_frame, text="High:")
        self.lbl_high_perc.pack(side=tk.LEFT, padx=(0,2))
        self.entry_high_perc = tk.Entry(low_high_frame, textvariable=self.merge_norm_high_perc_var, width=7)
        self.entry_high_perc.pack(side=tk.LEFT, padx=(0,0))
        tk.Label(merge_opts_frame, text="  ↳").grid(row=row_idx, column=0, sticky="e", padx=(20,2)) # Aligns with the checkbox
        self.merge_related_widgets_references.append((low_high_frame,)) # Note: only the frame is added here for simplicity. The sub-widgets are handled by parent state.
        self.widgets_to_disable_during_processing.extend([self.lbl_low_perc, self.entry_low_perc, self.lbl_high_perc, self.entry_high_perc]); row_idx += 1
        self.toggle_percentile_norm_options_active_state() # Call after creation

        # Alignment Method (Moved Output Format below this)
        lbl_merge_alignment = tk.Label(merge_opts_frame, text="Alignment Method:")
        lbl_merge_alignment.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.combo_merge_alignment = ttk.Combobox(merge_opts_frame, textvariable=self.merge_alignment_method_var, values=["Shift & Scale", "Linear Blend"], width=17, state="readonly")
        self.combo_merge_alignment.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        hl_mam = self._add_manual_help_icon(merge_opts_frame, "merge_alignment_method", row_idx, 2, sticky="w", padx=(0,5))
        self.merge_related_widgets_references.append((lbl_merge_alignment, self.combo_merge_alignment, hl_mam))
        self.widgets_to_disable_during_processing.extend([lbl_merge_alignment, self.combo_merge_alignment, hl_mam]); row_idx += 1
        
        # Output Format (Moved here)
        lbl_merge_fmt = tk.Label(merge_opts_frame, text="Output Format:")
        lbl_merge_fmt.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        merge_fmt_values = ["mp4", "main10_mp4", "png_sequence"] + (["exr_sequence", "exr"] if OPENEXR_AVAILABLE_GUI else [])
        self.combo_merge_fmt = ttk.Combobox(merge_opts_frame, textvariable=self.merge_output_format_var, values=merge_fmt_values, width=17, state="readonly")
        self.combo_merge_fmt.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        hl_mof = self._add_manual_help_icon(merge_opts_frame, "merge_output_format", row_idx, 2, sticky="w", padx=(0,5))
        self.merge_related_widgets_references.append((lbl_merge_fmt, self.combo_merge_fmt, hl_mof))
        self.widgets_to_disable_during_processing.extend([lbl_merge_fmt, self.combo_merge_fmt, hl_mof]); row_idx += 1

        # New: Output Suffix
        lbl_merge_suffix = tk.Label(merge_opts_frame, text="Output Suffix:")
        lbl_merge_suffix.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.entry_merge_suffix = tk.Entry(merge_opts_frame, textvariable=self.merge_output_suffix_var, width=18) # Match combobox width
        self.entry_merge_suffix.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        hl_mos = self._add_manual_help_icon(merge_opts_frame, "merge_output_suffix", row_idx, 2, sticky="w", padx=(0,5))
        self.merge_related_widgets_references.append((lbl_merge_suffix, self.entry_merge_suffix, hl_mos))
        self.widgets_to_disable_during_processing.extend([lbl_merge_suffix, self.entry_merge_suffix, hl_mos]); row_idx += 1
        
        # Merge Log Verbosity (remains last in this frame)
        lbl_merge_verbosity = tk.Label(merge_opts_frame, text="Merge Log Verbosity:")
        lbl_merge_verbosity.grid(row=row_idx, column=0, sticky="e", padx=5, pady=2)
        self.combo_merge_verbosity = ttk.Combobox(
            merge_opts_frame, 
            textvariable=self.merge_script_gui_silence_level_var, 
            values=gui_verbosity_options, 
            width=25, 
            state="readonly"
        )
        self.combo_merge_verbosity.grid(row=row_idx, column=1, padx=(0,2), pady=2, sticky="w")
        self.combo_merge_verbosity.bind("<<ComboboxSelected>>", self._update_gui_verbosity_from_combobox)
        hl_mlv = self._add_manual_help_icon(merge_opts_frame, "merge_log_verbosity", row_idx, 2, sticky="w", padx=(0,5))
        self.merge_related_widgets_references.append((lbl_merge_verbosity, self.combo_merge_verbosity, hl_mlv))
        self.widgets_to_disable_during_processing.extend([lbl_merge_verbosity, self.combo_merge_verbosity, hl_mlv]); row_idx += 1
        
        progress_bar_frame = tk.Frame(self.root)
        progress_bar_frame.pack(pady=(10, 0), padx=10, fill="x", expand=False)
        self.progress = ttk.Progressbar(progress_bar_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.pack(fill=tk.X, expand=True, padx=0, pady=0)

        ctrl_frame = tk.Frame(self.root)
        ctrl_frame.pack(pady=(5, 10), padx=10, fill="x", expand=False)

        start_frame = tk.Frame(ctrl_frame); start_frame.pack(side=tk.LEFT, padx=(0,2))
        self.start_button = tk.Button(start_frame, text="Start", command=self.start_thread, width=10)
        self.start_button.pack(side=tk.LEFT)

        cancel_frame = tk.Frame(ctrl_frame); cancel_frame.pack(side=tk.LEFT, padx=(2,2))
        self.cancel_button = tk.Button(cancel_frame, text="Cancel", command=self.stop_processing, width=10, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT)

        remerge_frame = tk.Frame(ctrl_frame); remerge_frame.pack(side=tk.LEFT, padx=(2,2))
        self.remerge_button = tk.Button(remerge_frame, text="Re-Merge Segments", command=self.re_merge_from_gui, width=18)
        self.remerge_button.pack(side=tk.LEFT)
        hl_rmb = tk.Label(remerge_frame, text="❓", fg="blue", cursor="hand2")
        hl_rmb.bind("<Button-1>", lambda e, key="remerge_button": self._show_help_for(key))
        hl_rmb.pack(side=tk.LEFT, padx=(1,0))

        genvis_frame = tk.Frame(ctrl_frame); genvis_frame.pack(side=tk.LEFT, padx=(2,2))
        self.generate_visuals_button = tk.Button(genvis_frame, text="Generate Seg Visuals", command=self.generate_segment_visuals_from_gui, width=20)
        self.generate_visuals_button.pack(side=tk.LEFT)
        hl_gvb = tk.Label(genvis_frame, text="❓", fg="blue", cursor="hand2")
        hl_gvb.bind("<Button-1>", lambda e, key="generate_visuals_button": self._show_help_for(key))
        hl_gvb.pack(side=tk.LEFT, padx=(1,0))

        self.clear_log_button = tk.Button(ctrl_frame, text="Clear Log", command=self._clear_log, width=10)
        self.clear_log_button.pack(side=tk.RIGHT, padx=(5,0))

        self.widgets_to_disable_during_processing.extend([
            self.start_button, self.remerge_button, hl_rmb,
            self.generate_visuals_button, hl_gvb, self.clear_log_button
        ])

        log_frame = tk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)
        self.log = tk.Text(log_frame, state="disabled", height=10, wrap=tk.WORD)
        log_scroll = tk.Scrollbar(log_frame, command=self.log.yview)
        self.log.config(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log.pack(side=tk.LEFT, fill="both", expand=True)
        self.toggle_merge_related_options_active_state()

    def _clear_log(self):
        if messagebox.askyesno("Clear Log", "Are you sure you want to clear the log?"):
            self.log.config(state="normal")
            self.log.delete(1.0, tk.END)
            self.log.config(state="disabled")
            global_log_message("GUI_LOG_CLEARED") # Use existing ID

    def _set_ui_processing_state(self, is_processing: bool):
        # ... (unchanged) ...
        new_state = tk.DISABLED if is_processing else tk.NORMAL
        cancel_state = tk.NORMAL if is_processing else tk.DISABLED
        unique_widgets = list(set(self.widgets_to_disable_during_processing))

        for widget in unique_widgets:
            if widget == self.cancel_button: continue
            if hasattr(widget, 'configure'):
                try:
                    if isinstance(widget, ttk.Combobox): widget.configure(state='disabled' if is_processing else 'readonly')
                    else: widget.configure(state=new_state)
                except tk.TclError: pass 
        
        if hasattr(self, 'cancel_button') and self.cancel_button:
             try: self.cancel_button.configure(state=cancel_state)
             except tk.TclError: pass

        if hasattr(self, 'file_menu'):
            try:
                for item_label in ["Load Settings...", "Save Settings As...", "Reset Settings to Default"]:
                    self.file_menu.entryconfig(item_label, state=new_state)
            except tk.TclError: pass

        self.toggle_merge_related_options_active_state()

    # ... (toggle_*_options_active_state methods unchanged) ...
    def toggle_keep_npz_dependent_options_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'keep_intermediate_npz_var') and hasattr(self, 'keep_npz_dependent_widgets')):
            return
        active = self.process_as_segments_var.get() and self.keep_intermediate_npz_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for widget in self.keep_npz_dependent_widgets:
            if hasattr(widget, 'configure'):
                try:
                    if isinstance(widget, ttk.Combobox): widget.configure(state='readonly' if active else 'disabled')
                    else: widget.configure(state=state)
                except tk.TclError: pass

    def toggle_dither_options_active_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'merge_dither_var')): return
        active = self.process_as_segments_var.get() and self.merge_dither_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for attr_name in ['lbl_dither_str', 'entry_dither_str']:
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                if widget and hasattr(widget, 'configure'):
                    try: widget.configure(state=state)
                    except tk.TclError: pass

    def toggle_gamma_options_active_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'merge_gamma_correct_var')): return
        active = self.process_as_segments_var.get() and self.merge_gamma_correct_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for attr_name in ['lbl_gamma_val', 'entry_gamma_val']:
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                if widget and hasattr(widget, 'configure'):
                    try: widget.configure(state=state)
                    except tk.TclError: pass

    def toggle_percentile_norm_options_active_state(self, *args):
        if not (hasattr(self, 'process_as_segments_var') and hasattr(self, 'merge_percentile_norm_var')): return
        active = self.process_as_segments_var.get() and self.merge_percentile_norm_var.get()
        state = tk.NORMAL if active else tk.DISABLED
        for attr_name in ['lbl_low_perc', 'entry_low_perc', 'lbl_high_perc', 'entry_high_perc']:
            if hasattr(self, attr_name):
                widget = getattr(self, attr_name)
                if widget and hasattr(widget, 'configure'):
                    try: widget.configure(state=state)
                    except tk.TclError: pass
    
    def toggle_merge_related_options_active_state(self, *args):
        if not hasattr(self, 'process_as_segments_var'): return
        active = self.process_as_segments_var.get()
        current_processing_state = tk.DISABLED
        if hasattr(self, 'start_button') and self.start_button and hasattr(self, 'cancel_button') and self.cancel_button:
            try:
                if self.start_button.cget('state') == tk.DISABLED and self.cancel_button.cget('state') == tk.NORMAL:
                    current_processing_state = tk.DISABLED
                else: current_processing_state = tk.NORMAL
            except tk.TclError: pass
        effective_state_for_merge_options = tk.DISABLED
        if current_processing_state == tk.NORMAL and active: effective_state_for_merge_options = tk.NORMAL
        if hasattr(self, 'merge_related_widgets_references'):
            for widget_tuple_or_item in self.merge_related_widgets_references:
                items_to_configure = widget_tuple_or_item if isinstance(widget_tuple_or_item, tuple) else (widget_tuple_or_item,)
                for widget_item in items_to_configure:
                    if hasattr(widget_item, 'configure'):
                        try:
                            if isinstance(widget_item, ttk.Combobox): widget_item.configure(state='readonly' if effective_state_for_merge_options == tk.NORMAL else 'disabled')
                            else: widget_item.configure(state=effective_state_for_merge_options)
                        except tk.TclError: pass
        if not active:
            if current_processing_state == tk.NORMAL:
                for var_attr_name in ['keep_intermediate_npz_var', 'merge_dither_var', 'merge_gamma_correct_var', 'merge_percentile_norm_var']:
                    if hasattr(self, var_attr_name):
                        var_to_set = getattr(self, var_attr_name)
                        if var_to_set: var_to_set.set(False)
        self.toggle_keep_npz_dependent_options_state()
        self.toggle_dither_options_active_state()
        self.toggle_gamma_options_active_state()
        self.toggle_percentile_norm_options_active_state()

    def add_param(self, parent, label, var, row):
        # ... (unchanged) ...
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=2)
        entry = tk.Entry(parent, textvariable=var, width=20)
        entry.grid(row=row, column=1, padx=5, pady=2, sticky="w")
        return entry

    def browse_input_folder(self): # Renamed from browse_input
        folder = filedialog.askdirectory(initialdir=self.input_dir_or_file_var.get())
        if folder:
            self.input_dir_or_file_var.set(os.path.normpath(folder))
            self.single_file_mode_active = False
            # Heuristic to check if selected folder is an image sequence itself
            if self._is_image_sequence_folder(folder):
                self.current_input_mode = "image_sequence_folder"
                global_log_message("GUI_INPUT_MODE_SET_IMG_SEQ_FOLDER", path=folder)
            else:
                self.current_input_mode = "batch_folder"
                global_log_message("GUI_INPUT_MODE_SET_BATCH_FOLDER", path=folder)
            # Make sure output dir is not inside input dir if it's a general batch folder
            # (This is advanced validation, can be added later)

    def browse_single_input_file(self): # New method
        filetypes = [("All Supported", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.gif *.png *.jpg *.jpeg *.bmp *.tiff *.exr"),
                     ("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.flv *.gif"),
                     ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.exr")]
        
        initial_dir_guess = self.input_dir_or_file_var.get()
        if os.path.isfile(initial_dir_guess): initial_dir_guess = os.path.dirname(initial_dir_guess)
        if not os.path.isdir(initial_dir_guess): initial_dir_guess = os.path.expanduser("~")


        filepath = filedialog.askopenfilename(initialdir=initial_dir_guess, filetypes=filetypes)
        if filepath:
            self.input_dir_or_file_var.set(os.path.normpath(filepath))
            self.single_file_mode_active = True
            ext = os.path.splitext(filepath)[1].lower()
            is_video = any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS)
            is_image = any(ext in img_ext.replace("*", "") for img_ext in self.IMAGE_EXTENSIONS)

            if is_video:
                self.current_input_mode = "single_video_file"
                global_log_message("GUI_INPUT_MODE_SET_SINGLE_VIDEO", path=filepath)
            elif is_image:
                self.current_input_mode = "single_image_file"
                global_log_message("GUI_INPUT_MODE_SET_SINGLE_IMAGE", path=filepath)
            else:
                global_log_message("GUI_INPUT_MODE_UNKNOWN_SINGLE_FILE_WARN", path=filepath)
                # Default to treating as video, or show error
                self.current_input_mode = "single_video_file" 
                messagebox.showwarning("Unknown File Type", f"Could not determine if '{os.path.basename(filepath)}' is a video or image. Assuming video.")

    def _is_image_sequence_folder(self, folder_path: str) -> bool:
        """Rudimentary check if a folder looks like an image sequence."""
        if not os.path.isdir(folder_path): return False
        
        image_files_count = 0
        video_files_count = 0
        sub_dirs_count = 0

        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                sub_dirs_count += 1
                continue # Don't check sub-sub-dirs for this simple heuristic
            
            ext = os.path.splitext(item)[1].lower()
            if any(ext in img_ext.replace("*", "") for img_ext in self.IMAGE_EXTENSIONS):
                image_files_count +=1
            elif any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS):
                video_files_count +=1
        
        # Heuristic: many images, no videos, no sub-directories
        return image_files_count > 5 and video_files_count == 0 and sub_dirs_count == 0

    def browse_output(self): # Unchanged
        folder = filedialog.askdirectory(initialdir=self.output_dir.get())
        if folder: self.output_dir.set(os.path.normpath(folder))

    # Old log_message removed, replaced by _queue_message_for_gui_log used as callback
    def process_queue(self):
        gui_log_updated_this_cycle = False
        while not self.message_queue.empty():
            try:
                msg_type, content = self.message_queue.get_nowait()
                
                # Make the debug print safer
                content_for_debug_print = str(content) # Convert to string first

                if msg_type == "log_display":
                    # Ensure content is a string before inserting for safety, though it should be.
                    if not isinstance(content, str):
                        # This would be an unexpected error if _queue_message_for_gui_log is correct
                        content = str(content) # Convert to string as a fallback

                    self.log.config(state="normal")
                    self.log.insert("end", f"{content}\n") 
                    self.log.config(state="disabled")
                    self.log.see("end")
                    gui_log_updated_this_cycle = True 
                elif msg_type == "progress":
                    self.progress["value"] = content # content is an int here
                elif msg_type == "set_ui_state":
                    self._set_ui_processing_state(content) # content is a bool here
            except queue.Empty:
                break
            except Exception as e:
                # This will catch the TypeError now if the above fix isn't enough, or other errors
                import traceback
                # print(traceback.format_exc()) # Print full traceback for errors within process_queue
        
        self.root.after(100, self.process_queue)

    def stop_processing(self):
        if self.processing_thread and self.processing_thread.is_alive():
            global_log_message("GUI_CANCEL_REQUEST") # Using existing ID
            self.stop_event.set()
        else: 
            global_log_message("GUI_NO_PROCESSING_TO_CANCEL_INFO") # New ID

    def start_thread(self):
        if self.processing_thread and self.processing_thread.is_alive():
            global_log_message("GUI_PROCESSING_ALREADY_RUNNING_WARN")
            return

        input_path_str = self.input_dir_or_file_var.get()
        if not input_path_str or not os.path.exists(input_path_str):
            global_log_message("GUI_INPUT_PATH_EMPTY_ERROR")
            messagebox.showerror("Error", f"Input path does not exist: {input_path_str}")
            return
        
        # === Add path analysis here ===
        # Re-determine mode based on the current text in the input field
        # This makes it robust to typed paths or loaded configs where browse buttons weren't the last action.
        determined_mode, determined_single_source = self._determine_input_mode_from_path(input_path_str)
        
        # Update the GUI's state based on this fresh determination
        self.current_input_mode = determined_mode
        self.single_file_mode_active = determined_single_source
        # === End of added path analysis ===


        if not os.path.exists(input_path_str): # Check existence *after* trying to determine mode (as _determine... also checks)
            global_log_message("GUI_INPUT_PATH_INVALID_ERROR", path=input_path_str)
            messagebox.showerror("Error", f"Input path does not exist: {input_path_str}")
            return

        sources_to_process_specs = [] # List of dicts: {"path": str, "type": str, "basename": str}

        if self.single_file_mode_active: # Now this flag is reliably set
            self.effective_move_original_on_completion = False
            basename = ""
            # For image_sequence_folder, basename is folder name. For files, it's filename without ext.
            if self.current_input_mode == "image_sequence_folder":
                basename = os.path.basename(input_path_str)
            else: # single_video_file or single_image_file
                basename = os.path.splitext(os.path.basename(input_path_str))[0]
            
            sources_to_process_specs.append({
                "path": input_path_str,
                "type": self.current_input_mode, 
                "basename": basename
            })
        else: # Batch folder mode (self.current_input_mode will be "batch_folder")
            self.effective_move_original_on_completion = self.MOVE_ORIGINAL_TO_FINISHED_FOLDER_ON_COMPLETION
            if self.current_input_mode == "batch_folder": # Explicit check
                # Scan the batch folder for videos and image sequence subfolders
                try:
                    for item_name in os.listdir(input_path_str):
                        item_full_path = os.path.join(input_path_str, item_name)
                        # ... (rest of the batch folder scanning logic from previous version)
                        if os.path.isfile(item_full_path):
                            ext = os.path.splitext(item_name)[1].lower()
                            if any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS):
                                basename = os.path.splitext(item_name)[0]
                                sources_to_process_specs.append({
                                    "path": item_full_path,
                                    "type": "video_file", # When found in batch, it's a standard video_file
                                    "basename": basename
                                })
                        elif os.path.isdir(item_full_path):
                            if self._is_image_sequence_folder(item_full_path):
                                basename = item_name 
                                sources_to_process_specs.append({
                                    "path": item_full_path,
                                    "type": "image_sequence_folder",
                                    "basename": basename
                                })
                except NotADirectoryError: # Should be caught by _determine_input_mode_from_path if path was invalid
                    global_log_message("GUI_INPUT_PATH_NOT_DIR_FOR_BATCH_ERROR", path=input_path_str)
                    messagebox.showerror("Error", f"Input path is not a directory for batch processing: {input_path_str}")
                    return
                except OSError as e: # Catch other potential OS errors from listdir
                    global_log_message("GUI_LISTDIR_OS_ERROR", path=input_path_str, error=str(e))
                    messagebox.showerror("Error", f"Could not read directory contents for '{input_path_str}':\n{e}")
                    return
            else: # Should not happen if _determine_input_mode_from_path is correct
                global_log_message("GUI_START_THREAD_UNEXPECTED_MODE_AFTER_DETERMINATION", mode=self.current_input_mode, path=input_path_str)
                messagebox.showerror("Internal Error", f"Unexpected input mode '{self.current_input_mode}' for path '{input_path_str}'. Please report this.")
                return


        if not sources_to_process_specs:
            global_log_message("GUI_NO_VALID_SOURCES_FOUND", path_scanned=input_path_str, mode=self.current_input_mode)
            return
        
        final_jobs_to_process = []
        base_job_info_map_for_run = {} 

        gui_fps_setting = self.target_fps.get()
        gui_len_setting = self.process_length.get()
        gui_win_setting = self.window_size.get()
        gui_ov_setting = self.overlap.get()

        for source_spec in sources_to_process_specs:
            source_path = source_spec["path"]
            current_gui_mode = source_spec["type"] 
            base_name = source_spec["basename"]

            source_type_for_define = ""
            # ... (mapping current_gui_mode to source_type_for_define) ...
            if current_gui_mode == "single_video_file" or current_gui_mode == "video_file":
                source_type_for_define = "video_file"
            elif current_gui_mode == "image_sequence_folder":
                source_type_for_define = "image_sequence_folder"
            elif current_gui_mode == "single_image_file":
                source_type_for_define = "single_image_file"
            # ...

            # num_frames_for_single_img_param = -1 # Default for define_video_segments
            # if source_type_for_define == "single_image_file":
            #     num_frames_for_single_img_param = DepthCrafterGUI.DEFAULT_SINGLE_IMAGE_FRAME_COUNT # Use the class attribute

            all_potential_segments_for_video, base_job_info_initial = define_video_segments(
                video_path_or_folder=source_path,
                original_basename=base_name,
                gui_target_fps_setting=gui_fps_setting,
                gui_process_length_overall=gui_len_setting,
                gui_segment_output_window_frames=gui_win_setting,
                gui_segment_output_overlap_frames=gui_ov_setting,
                source_type=source_type_for_define, # The mapped type for define_video_segments internal logic
            )

            if not base_job_info_initial:
                global_log_message("PROCESSING_SKIPPED", item_name=base_name, reason="Issues in segment definition (metadata error).")
                continue
            
            base_job_info_map_for_run[source_path] = base_job_info_initial.copy() # Keyed by actual source path

            if self.process_as_segments_var.get():
                if not all_potential_segments_for_video:
                    reason_skip = "Too short or invalid overlap/settings" if base_job_info_initial.get("original_video_raw_frame_count", 0) > 0 else "Source issue or zero frames/duration"
                    global_log_message("GUI_NO_SEGMENTS_DEFINED_SKIP", basename=base_name, reason=reason_skip)
                    continue

                segment_subfolder_name = get_segment_output_folder_name(base_name)
                segment_subfolder_path = os.path.join(self.output_dir.get(), segment_subfolder_name)
                current_video_base_info_ref = base_job_info_map_for_run[source_path]
                
                segments_for_this_video, action_taken = self._get_segments_to_resume_or_overwrite(
                    source_path, base_name, segment_subfolder_path, all_potential_segments_for_video,
                    current_video_base_info_ref # This ref will be updated with pre_existing_successful_jobs
                )
                global_log_message("GUI_RESUME_ACTION", video_name=base_name, action_taken=action_taken, num_segments=len(segments_for_this_video))
                if segments_for_this_video:
                    final_jobs_to_process.extend(segments_for_this_video) # these jobs now contain full base_job_info
            
            else: # Full "video" processing (could be actual video, sequence, or single image made into clip)
                # Output path check for full processing
                # The actual extension depends on the source. For now, assume mp4 for this check.
                # get_full_video_output_filename is generic.
                full_out_check_path = os.path.join(self.output_dir.get(), get_full_video_output_filename(base_name, "mp4"))
                proceed_full = True
                if os.path.exists(full_out_check_path): # Simplified check, might need to be smarter based on source type
                    if not messagebox.askyesno("Overwrite?", f"An output file for '{base_name}' might exist (e.g., MP4):\n{full_out_check_path}\n\nOverwrite if it exists?"):
                        global_log_message("GUI_FULL_VIDEO_SKIP_NO_OVERWRITE", basename=base_name)
                        proceed_full = False
                
                if proceed_full:
                    # Construct a single job representing the "full" processing of the source
                    # This job needs all info from base_job_info_initial
                    full_source_job = {
                        **base_job_info_initial, # Contains path, type, basename, fps, frame_count, gui_fps_setting
                        "is_segment": False,
                        # These GUI settings are for reference if needed by backend, though backend uses its own params for full run
                        "gui_desired_output_window_frames": gui_win_setting, 
                        "gui_desired_output_overlap_frames": gui_ov_setting 
                    }
                    final_jobs_to_process.append(full_source_job)

        if final_jobs_to_process:
            self._set_ui_processing_state(True)
            self.processing_thread = threading.Thread(target=self._start_processing_wrapper, 
                                                      args=(final_jobs_to_process, base_job_info_map_for_run), 
                                                      daemon=True)
            self.processing_thread.start()
        else:
            global_log_message("GUI_NO_JOBS_TO_PROCESS_FINAL_INFO")

    def _start_processing_wrapper(self, video_processing_jobs, base_job_info_map):
        try: 
            self.start_processing(video_processing_jobs, base_job_info_map)
        finally: 
            self.message_queue.put(("set_ui_state", False)) 

    def _determine_input_mode_from_path(self, path_str: str) -> Tuple[str, bool]:
        """
        Analyzes a path string and determines the input mode and if it's a single source.
        Returns: (input_mode_str, is_single_source_bool)
        """
        if not path_str or not os.path.exists(path_str):
            global_log_message("GUI_INPUT_PATH_INVALID_FOR_MODE_DETECT", path=path_str)
            return "batch_folder", False # Default to batch if path is invalid

        is_single_source = False
        mode = "batch_folder" # Default

        if os.path.isfile(path_str):
            is_single_source = True
            ext = os.path.splitext(path_str)[1].lower()
            is_video = any(ext in vid_ext.replace("*", "") for vid_ext in self.VIDEO_EXTENSIONS)
            is_image = any(ext in img_ext.replace("*", "") for img_ext in self.IMAGE_EXTENSIONS)

            if is_video:
                mode = "single_video_file"
            elif is_image:
                mode = "single_image_file"
            else:
                # Unknown file type, treat as if it's a folder or log error
                global_log_message("GUI_INPUT_MODE_UNKNOWN_TYPED_FILE_WARN", path=path_str)
                mode = "batch_folder" # Fallback to prevent error, or could be an error state
                is_single_source = False # Revert, as we can't process it as a single known file
        elif os.path.isdir(path_str):
            # If it's a directory, it could be a batch folder or an image sequence folder.
            # The _is_image_sequence_folder heuristic is used.
            if self._is_image_sequence_folder(path_str):
                mode = "image_sequence_folder"
                is_single_source = True # An image sequence folder is treated as a single processing unit
            else:
                mode = "batch_folder"
                is_single_source = False
        else:
            # Path exists but is not a file or directory (e.g., symlink to nowhere, etc.)
            global_log_message("GUI_INPUT_PATH_NOT_FILE_OR_DIR_WARN", path=path_str)
            mode = "batch_folder" # Default/fallback
            is_single_source = False
            
        global_log_message("GUI_INPUT_MODE_DETERMINED", path=path_str, mode=mode, is_single=is_single_source)
        return mode, is_single_source

    def save_config(self):
        config = self._collect_all_settings() # This should now get input_dir_or_file_var
        config[self.LAST_SETTINGS_DIR_CONFIG_KEY] = self.last_settings_dir
        
        # Add current_input_mode and single_file_mode_active to be saved
        config["current_input_mode"] = self.current_input_mode 
        config["single_file_mode_active"] = self.single_file_mode_active
        
        try:
            with open(self.CONFIG_FILENAME, "w") as f: json.dump(config, f, indent=4)
            # No need to log here, GUI might be closing
        except Exception as e: 
            print(f"Warning (GUI save_config): Could not save config: {e}")

    def load_config(self):
        if os.path.exists(self.CONFIG_FILENAME):
            try:
                with open(self.CONFIG_FILENAME, "r") as f: config = json.load(f)
                loaded_settings_for_tkvars = {k: v for k, v in config.items() if k in self.all_tk_vars}
                for key, value in loaded_settings_for_tkvars.items():
                    if key in self.all_tk_vars:
                        try: self.all_tk_vars[key].set(value)
                        except tk.TclError: 
                            print(f"Warning (GUI load_config): Could not set var {key} during early config load.")
                
                self.last_settings_dir = config.get(self.LAST_SETTINGS_DIR_CONFIG_KEY, os.getcwd())
                
                # Load current_input_mode and single_file_mode_active
                self.current_input_mode = config.get("current_input_mode", "batch_folder") # Default if not found
                self.single_file_mode_active = config.get("single_file_mode_active", False) # Default if not found
                
                # Update GUI verbosity from loaded config if applicable
                set_gui_verbosity(self._get_mapped_gui_verbosity_level())
                global_log_message("GUI_CONFIG_LOADED_INFO", filename=self.CONFIG_FILENAME) # Log after successful load
            except Exception as e:
                print(f"Warning (GUI load_config): Could not load config '{self.CONFIG_FILENAME}': {e}")
                # Set defaults if loading fails for these crucial state vars
                self.last_settings_dir = os.getcwd()
                self.current_input_mode = "batch_folder"
                self.single_file_mode_active = False
        else: 
            self.last_settings_dir = os.getcwd()
            self.current_input_mode = "batch_folder" # Set defaults if no config file
            self.single_file_mode_active = False
            global_log_message("GUI_CONFIG_NOT_FOUND_INFO", filename=self.CONFIG_FILENAME)

    def on_close(self):
        self.save_config()
        if self.processing_thread and self.processing_thread.is_alive():
            global_log_message("GUI_STOPPING_ON_CLOSE_INFO") # New ID
            self.stop_event.set()
            self.processing_thread.join(timeout=10)
            if self.processing_thread.is_alive(): 
                global_log_message("GUI_THREAD_FORCE_EXIT_WARN") # New ID
        # Clean up message_catalog callback to prevent issues if app is somehow re-instantiated
        set_gui_logger_callback(None)
        self.root.destroy()

    def re_merge_from_gui(self):
        if not merge_depth_segments: 
            messagebox.showerror("Error", "Merge module not available."); return
        meta_file = filedialog.askopenfilename(title="Select Master Metadata JSON for Re-Merging", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialdir=self.output_dir.get())
        if not meta_file: return
        
        base_name_from_meta = os.path.splitext(os.path.basename(meta_file))[0].replace("_master_meta", "")
        output_suffix = self.merge_output_suffix_var.get() # Get suffix
        remerge_base_name = f"{base_name_from_meta}{output_suffix}" # Use suffix for base name

        out_fmt = self.merge_output_format_var.get()
        
        def_ext_fmt = out_fmt
        if out_fmt == "main10_mp4": # Covers "main10_mp4"
            def_ext_fmt = "mp4"
        elif out_fmt in ["png_sequence", "exr_sequence"]:
            def_ext_fmt = "" # No extension for sequence folders
        elif out_fmt == "exr":
            def_ext_fmt = "exr"

        def_ext = f".{def_ext_fmt}" if def_ext_fmt else ""

        ftypes_map = {
            "mp4": [("MP4 (H.264 8-bit)", "*.mp4")],
            "main10_mp4": [("MP4 (HEVC 10-bit)", "*.mp4")], # New entry
            "png_sequence": [("PNG Seq (Select Folder)", "")],
            "exr_sequence": [("EXR Seq (Select Folder)", "")],
            "exr": [("EXR File", "*.exr")]
        }
        curr_ftypes = ftypes_map.get(out_fmt, []) + [("All files", "*.*")]
        out_path = None

        if "sequence" in out_fmt:
            parent_dir = filedialog.askdirectory(title=f"Select Parent Dir for Re-Merged {out_fmt.upper()} Sequence...", initialdir=self.output_dir.get())
            if parent_dir: out_path = parent_dir # merge_depth_segments will create subfolder using remerge_base_name
        else:
            # Construct the initial filename based on YOUR variables
            initial_filename_for_dialog_actual = f"{remerge_base_name}{def_ext}"

            out_path = filedialog.asksaveasfilename(
                title=f"Save Re-Merged {out_fmt.upper()} As...", 
                initialdir=self.output_dir.get(), 
                initialfile=f"{remerge_base_name}{def_ext}", # Filename includes suffix
                defaultextension=def_ext, 
                filetypes=curr_ftypes
            )

        if not out_path: 
            global_log_message("GUI_REMERGE_CANCELLED_NO_PATH"); return
            
        align_method = "linear_blend" if self.merge_alignment_method_var.get() == "Linear Blend" else "shift_scale"
        
        args = {"master_meta_path": meta_file, "output_path_arg": out_path,
                "do_dithering": self.merge_dither_var.get(), "dither_strength_factor": self.merge_dither_strength_var.get(),
                "apply_gamma_correction": self.merge_gamma_correct_var.get(), "gamma_value": self.merge_gamma_value_var.get(),
                "use_percentile_norm": self.merge_percentile_norm_var.get(), "norm_low_percentile": self.merge_norm_low_perc_var.get(),
                "norm_high_percentile": self.merge_norm_high_perc_var.get(), "output_format": out_fmt,
                "merge_alignment_method": align_method,
                "output_filename_override_base": remerge_base_name} # Pass the suffixed base name

        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Another process is running. Please wait."); return
        
        global_log_message("GUI_ACTION_STARTED", action_name="Re-Merge", target_name=os.path.basename(meta_file))
        self._set_ui_processing_state(True)
        self.processing_thread = threading.Thread(target=self._execute_re_merge_wrapper, args=(args,), daemon=True); self.processing_thread.start()

    def _execute_re_merge_wrapper(self, remerge_args_dict):
        try: self._execute_re_merge(remerge_args_dict)
        finally: self.message_queue.put(("set_ui_state", False))

    def _execute_re_merge(self, remerge_args_dict):
        self.stop_event.clear(); self.progress["value"] = 0; self.progress["maximum"] = 1
        # Logged by caller: GUI_ACTION_STARTED
        start_time = time.perf_counter()
        try:
            if merge_depth_segments:
                # merge_depth_segments.set_gui_logger_callback removed
                saved_path = merge_depth_segments.merge_depth_segments(**remerge_args_dict)
                # merge_depth_segments logs its own success
            else: global_log_message("MERGE_MODULE_UNAVAILABLE", video_name="N/A for re-merge action")
        except Exception as e:
            global_log_message("GUI_REMERGE_EXEC_ERROR", error=str(e), traceback_info=sys.exc_info()) # New ID
        finally:
            # merge_depth_segments.set_gui_logger_callback(None) removed
            duration = format_duration(time.perf_counter() - start_time)
            global_log_message("GUI_ACTION_COMPLETE", action_name="Re-Merge", target_name=os.path.basename(remerge_args_dict['master_meta_path']), duration=duration)
            self.message_queue.put(("progress", 1))

    def generate_segment_visuals_from_gui(self):
        if self.processing_thread and self.processing_thread.is_alive():
            messagebox.showwarning("Busy", "Another process is running. Please wait."); return
        meta_file = filedialog.askopenfilename(title="Select Master Metadata JSON for Segment Visual Generation", filetypes=[("JSON files", "*.json"), ("All files", "*.*")], initialdir=self.output_dir.get())
        if not meta_file: 
            global_log_message("GUI_GEN_VISUALS_CANCELLED_NO_META") # New ID
            return
        vis_fmt = self.keep_intermediate_segment_visual_format_var.get()
        if vis_fmt == "none": 
            messagebox.showinfo("Info", "Segment Visual Format is 'none'. Select a valid format."); return
        if not messagebox.askyesno("Generate/Overwrite Visuals?", f"Generate '{vis_fmt}' visuals for segments in '{os.path.basename(meta_file)}'?\nThis may overwrite existing visuals."):
            global_log_message("GUI_GEN_VISUALS_CANCELLED_BY_USER") # New ID
            return
        args = {"master_meta_path": meta_file, "visual_format_to_generate": vis_fmt}
        global_log_message("GUI_ACTION_STARTED", action_name="Segment Visual Generation", target_name=f"{os.path.basename(meta_file)} (Format: {vis_fmt})")
        self._set_ui_processing_state(True)
        self.processing_thread = threading.Thread(target=self._execute_generate_segment_visuals_wrapper, args=(args,), daemon=True); self.processing_thread.start()

    def _execute_generate_segment_visuals_wrapper(self, gen_visual_args_dict):
        try: self._execute_generate_segment_visuals(gen_visual_args_dict)
        finally: self.message_queue.put(("set_ui_state", False))

    def _execute_generate_segment_visuals(self, gen_visual_args_dict):
        self.stop_event.clear(); self.progress["value"] = 0
        master_path = gen_visual_args_dict["master_meta_path"]
        vis_fmt = gen_visual_args_dict["visual_format_to_generate"]
        start_time = time.perf_counter()
        
        meta_data = load_json_file(master_path) 
        if not meta_data: return 
        
        jobs = [j for j in meta_data.get("jobs_info", []) if j.get("status") == "success" and j.get("output_segment_filename")]
        if not jobs: 
            global_log_message("GUI_GEN_VISUALS_NO_SUCCESSFUL_SEGS", master_file=os.path.basename(master_path)) 
            return
        self.progress["maximum"] = len(jobs)
        seg_folder_path = os.path.dirname(master_path)
        updated_visual_paths = {}

        for i, job_meta in enumerate(jobs):
            if self.stop_event.is_set(): 
                global_log_message("GUI_GEN_VISUALS_CANCELLED_MID_PROCESS") 
                break
            seg_id, npz_name = job_meta.get("segment_id"), job_meta.get("output_segment_filename")
            npz_path = os.path.join(seg_folder_path, npz_name)
            global_log_message("GUI_GEN_VISUALS_PROCESSING_SEGMENT", segment_id=seg_id + 1 if seg_id is not None else '?', 
                               total_segments=len(jobs), npz_name=npz_name, format=vis_fmt) 
            
            if not os.path.exists(npz_path): 
                global_log_message("FILE_NOT_FOUND", filepath=npz_path)
                continue
            try:
                with np.load(npz_path) as data:
                    if 'frames' not in data.files: 
                        global_log_message("NPZ_LOAD_KEY_ERROR", key='frames', filepath=npz_name)
                        continue
                    raw_frames = data['frames']
                if raw_frames.size == 0: 
                    global_log_message("GUI_GEN_VISUALS_SEGMENT_EMPTY_WARN", npz_name=npz_name) 
                    continue
                
                norm_frames = (raw_frames - raw_frames.min()) / (raw_frames.max() - raw_frames.min()) if raw_frames.max() != raw_frames.min() else np.zeros_like(raw_frames)
                norm_frames = np.clip(norm_frames, 0, 1)
                base_name_no_ext = os.path.splitext(npz_name)[0]
                save_path, save_err = None, None
                fps = float(job_meta.get("processed_at_fps", meta_data.get("original_video_details", {}).get("original_fps", 30.0)))
                if fps <= 0: fps = 30.0

                if vis_fmt == "mp4" or vis_fmt == "main10_mp4": # Modified to handle main10_mp4
                    # Pass vis_fmt as output_format to the util
                    save_path, save_err = save_depth_visual_as_mp4_util(
                        norm_frames, 
                        os.path.join(seg_folder_path, f"{base_name_no_ext}_visual.mp4"), # Filename extension is still .mp4
                        fps,
                        output_format=vis_fmt # Pass "mp4" or "main10_mp4"
                    )
                elif vis_fmt == "png_sequence":
                    save_path, save_err = save_depth_visual_as_png_sequence_util(norm_frames, seg_folder_path, base_name_no_ext)
                elif vis_fmt == "exr_sequence":
                     if OPENEXR_AVAILABLE_GUI: save_path, save_err = save_depth_visual_as_exr_sequence_util(norm_frames, seg_folder_path, base_name_no_ext)
                     else: save_err = "OpenEXR module not available in GUI environment."
                elif vis_fmt == "exr":
                    if OPENEXR_AVAILABLE_GUI:
                        first_frame = norm_frames[0] if len(norm_frames) > 0 else None
                        if first_frame is None: save_err = "No frame data for single EXR."
                        else: save_path, save_err = save_depth_visual_as_single_exr_util(first_frame, seg_folder_path, base_name_no_ext)
                    else: save_err = "OpenEXR module not available in GUI environment."

                if save_path:
                    global_log_message("GUI_GEN_VISUALS_SAVE_SUCCESS", path=save_path) 
                    if seg_id is not None: updated_visual_paths[seg_id] = {"path": os.path.abspath(save_path), "format": vis_fmt}
                if save_err: 
                    global_log_message("GUI_GEN_VISUALS_SAVE_ERROR", npz_name=npz_name, error=save_err, format_requested=vis_fmt) 
            except Exception as e:
                global_log_message("GUI_GEN_VISUALS_SEGMENT_PROCESSING_ERROR", npz_name=npz_name, error=str(e), traceback_info=sys.exc_info()) 
            self.message_queue.put(("progress", i + 1))
        
        if updated_visual_paths:
            global_log_message("GUI_GEN_VISUALS_UPDATING_MASTER_META") # New ID
            meta_content_update = load_json_file(master_path) # Util logs
            if meta_content_update:
                updated_count = 0
                for job_entry in meta_content_update.get("jobs_info", []):
                    s_id = job_entry.get("segment_id")
                    if s_id in updated_visual_paths:
                        job_entry["intermediate_visual_path"] = updated_visual_paths[s_id]["path"]
                        job_entry["intermediate_visual_format_saved"] = updated_visual_paths[s_id]["format"]
                        updated_count +=1
                if updated_count > 0:
                    if save_json_file(meta_content_update, master_path, indent=4): # Util logs
                         global_log_message("GUI_GEN_VISUALS_MASTER_META_UPDATE_SUCCESS", count=updated_count) # New ID
                    # else save_json_file logs its own error
                else: global_log_message("GUI_GEN_VISUALS_MASTER_META_NO_UPDATES_NEEDED") # New ID
            # else: load_json_file logs its own error
        
        duration = format_duration(time.perf_counter() - start_time)
        global_log_message("GUI_ACTION_COMPLETE", action_name="Segment Visual Generation", 
                           target_name=f"{os.path.basename(master_path)} (Format: {vis_fmt})", duration=duration)
        self.message_queue.put(("progress", len(jobs)))

    def _get_mapped_gui_verbosity_level(self):
        level_str = self.merge_script_gui_silence_level_var.get()
        if level_str == "Verbose (Detail)":       return message_catalog.VERBOSITY_LEVEL_DETAIL   # 15
        elif level_str == "Normal (Info)":        return message_catalog.VERBOSITY_LEVEL_INFO     # 20
        elif level_str == "Less Verbose (Warnings)":return message_catalog.VERBOSITY_LEVEL_WARNING  # 30
        elif level_str == "Silent (Errors Only)": return message_catalog.VERBOSITY_LEVEL_ERROR    # 40
        return message_catalog.VERBOSITY_LEVEL_INFO

    def _update_gui_verbosity_from_combobox(self, event=None):
        new_level = self._get_mapped_gui_verbosity_level()
        message_catalog.set_gui_verbosity(new_level) # Use module prefix for the function call too for consistency
        global_log_message("GUI_VERBOSITY_CHANGED", level_name=self.merge_script_gui_silence_level_var.get(), numeric_level=new_level)


if __name__ == "__main__":
    root = tk.Tk()
    # Console verbosity for Tkinter/internal Python stuff if not handled by GUI log
    # from message_catalog import set_console_verbosity 
    # set_console_verbosity(VERBOSITY_LEVEL_WARNING) # Example
    app = DepthCrafterGUI(root)
    root.mainloop()