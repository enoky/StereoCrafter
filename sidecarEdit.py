import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import json
import glob
import math
import _tkinter
from moviepy.editor import VideoFileClip

class SidecarEditTool:
    PARAMETER_CONFIG = {
        # Key: {Label, Type, Default, Min, Max, Decimals, FusionKey(fsexport), SidecarKey(fssidecar), Group}
        "convergence": {
            "label": "Convergence Plane", "type": float, "default": 0.5, "min": 0.0, "max": 1.0, 
            "decimals": 3, "fusion_key": "Convergence", "sidecar_key": "convergence_plane"
        },
        "max_disparity": {
            "label": "Max Disparity", "type": float, "default": 35.0, "min": 0.0, "max": 100.0, 
            "decimals": 1, "fusion_key": "MaxDisparity", "sidecar_key": "max_disparity"
        },
        "gamma": {
            "label": "Gamma Correction", "type": float, "default": 1.0, "min": 0.1, "max": 5.0,
            "decimals": 2, "fusion_key": "FrontGamma", "sidecar_key": "gamma"
        },
        "frame_overlap": {
            "label": "Frame Overlap", "type": int, "default": 3, "min": 0, "max": 60,
            "decimals": 0, "fusion_key": "Overlap", "sidecar_key": "frame_overlap"
        },
        "input_bias": {
            "label": "Input Bias", "type": float, "default": 0, "min": 0.0, "max": 1.0, 
            "decimals": 2, "fusion_key": "Bias", "sidecar_key": "input_bias"
        }
    }
    # Set the maximum number of fields to pre-create
    MAX_PARAMETER_FIELDS = 8 
    # ---------------------------------------

    def __init__(self, root):
        self.root = root
        self.root.title("SidecarEdit - Depthmap Parameters")
        # self.root.geometry("800x650")

        # --- Step 1: Initialize all Tkinter control variables and basic data structures ---
        self.current_folder_var = tk.StringVar(value="")
        self.recent_folders_list = []
        self.MAX_RECENT_FOLDERS = 10

        self.current_filename_var = tk.StringVar(value="No file loaded")
        self.file_status_var = tk.StringVar(value="0 of 0")
        self.jump_to_file_var = tk.StringVar(value="1")
        self.status_message_var = tk.StringVar(value="Ready.")
        
        # ADD START: Dynamic Parameter Variables
        self.param_vars = {}        # Holds tk.DoubleVar/IntVar instances keyed by "convergence", "max_disparity", etc.
        self.param_entry_widgets = {} # Holds the tk.Entry widgets
        self.param_label_widgets = {} # Holds the tk.Label widgets
        self.param_vcmds = {}       # Holds the validation commands
        self.active_params = []     # List of currently active parameter keys (e.g., ["convergence", "max_disparity", "gamma"])
        
        self.all_depth_map_files = []
        self.current_file_index = -1
        
        self.progress_save_data = {}

        self.config_file_path = "config_sidecaredit.json"
        self.progress_file_path = None

        # --- Step 2: Load Configuration (this doesn't interact with GUI directly yet) ---
        self._load_gui_config()

        # --- Step 3: Create all GUI widgets and menus ---
        self._create_menu()
        self._create_widgets() # <-- ALL WIDGETS ARE NOW GUARANTEED TO BE CREATED HERE
        
        # --- Step 6: Perform Initial Data Load (This should be the very last step in __init__) ---
        # This call is now safe because all GUI elements are initialized.
        if self.recent_folders_list:
            self.current_folder_var.set(self.recent_folders_list[0])
            self._load_depth_maps()

    def _add_to_recent_folders(self, folder_path):
        if folder_path in self.recent_folders_list:
            self.recent_folders_list.remove(folder_path)
        self.recent_folders_list.insert(0, folder_path)
        self._trim_recent_folders()
        self._update_recent_folders_menu()
        self._save_gui_config()

    def _browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.current_folder_var.set(folder_selected)
            self._add_to_recent_folders(folder_selected)
            self._load_depth_maps()

    def _create_dynamic_parameter_fields(self):
        # Clear old widgets just in case
        for widget in self.params_container_frame.winfo_children():
            widget.destroy()

        param_keys = list(self.PARAMETER_CONFIG.keys())
        
        for i in range(self.MAX_PARAMETER_FIELDS):
            # Use 'i' for row index and determine the key to use
            param_key = param_keys[i] if i < len(param_keys) else f"dummy_{i}"
            config = self.PARAMETER_CONFIG.get(param_key)
            
            # 1. Create Variable, Validation Command, and trace
            if config:
                var = tk.DoubleVar(value=config["default"]) if config["type"] is float else tk.IntVar(value=config["default"])
                
                # We need a partial function to capture the current param_key for validation
                vcmd_func = self.root.register(lambda p, k=param_key: self._validate_dynamic_input(p, k))
                self.param_vcmds[param_key] = vcmd_func
                
                # Trace must call a generic update function with the key
                var.trace_add("write", lambda *args, k=param_key: self._update_entry_display(k))
                self.param_vars[param_key] = var
                
                # 2. Create Label
                label_text = f"{config['label']} ({config['min']}-{config['max']}, {config['type'].__name__}):"
                label = ttk.Label(self.params_container_frame, text=label_text)
                label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
                self.param_label_widgets[param_key] = label

                # 3. Create Entry
                entry = ttk.Entry(
                    self.params_container_frame,
                    textvariable=var,
                    width=15,
                    validate="focusout",
                    validatecommand=(vcmd_func, '%P')
                )
                entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
                entry.bind("<Return>", lambda e, k=param_key: self._validate_dynamic_input(self.param_vars[k].get(), k, is_return=True))
                self.param_entry_widgets[param_key] = entry
                
                # Initial display update (which happens via trace_add right above)
                self._update_entry_display(param_key)
            else:
                # Placeholder for unused fields (e.g., fields 6, 7, 8)
                # We still create dummy widgets so the grid rows are reserved
                dummy_label = ttk.Label(self.params_container_frame, text="")
                dummy_label.grid(row=i, column=0, sticky="w", padx=5, pady=2)
                self.param_label_widgets[param_key] = dummy_label

                dummy_entry = ttk.Entry(self.params_container_frame, width=15, state="disabled")
                dummy_entry.grid(row=i, column=1, sticky="w", padx=5, pady=2)
                self.param_entry_widgets[param_key] = dummy_entry

    def _create_menu(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Load Depth Map Folder...", command=self._browse_folder)
        
        self.recent_folders_submenu = tk.Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Recent Folders", menu=self.recent_folders_submenu)
        self._update_recent_folders_menu()

        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save Current File Progress", command=self._save_progress)
        self.file_menu.add_command(label="Load Fusion Export File (.fsexport)...", command=self._load_fusion_export_file)
        self.file_menu.add_command(label="Generate All Sidecar FSSIDECARs...", command=self._generate_sidecar_jsons)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self._exit_app)

    def _create_widgets(self):
        folder_frame = ttk.LabelFrame(self.root, text="Depth Map Folder")
        folder_frame.pack(pady=10, padx=10, fill="x")
        
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.current_folder_var, width=30)
        self.folder_entry.grid(row=0, column=0, padx=5, pady=5, sticky="ew") # Use grid for better control within labelframe
        folder_frame.grid_columnconfigure(0, weight=1) # Make entry expand
        self.folder_entry.bind("<Return>", self._load_folder_from_entry)
        self.folder_entry.bind("<FocusOut>", self._load_folder_from_entry)

        ttk.Button(folder_frame, text="Browse", command=self._browse_folder).grid(row=0, column=1, padx=5, pady=5)

        file_info_frame = ttk.LabelFrame(self.root, text="Current Depth Map")
        file_info_frame.pack(pady=5, padx=10, fill="x")

        ttk.Label(file_info_frame, text="Filename:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(file_info_frame, textvariable=self.current_filename_var, wraplength=400).grid(row=0, column=1, sticky="w", padx=5, pady=2)
        file_info_frame.grid_columnconfigure(1, weight=1) # Make filename label expand
        ttk.Label(file_info_frame, text="File Status:").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(file_info_frame, textvariable=self.file_status_var).grid(row=1, column=1, sticky="w", padx=5, pady=2)

        # DYNAMIC PARAMETERS FRAME
        self.params_container_frame = ttk.LabelFrame(self.root, text="Dynamic Parameters")
        self.params_container_frame.pack(pady=5, padx=10, fill="x")
        self.params_container_frame.grid_columnconfigure(1, weight=1) # Allow parameter entry to expand
        
        # Create all possible parameter fields (up to MAX_PARAMETER_FIELDS)
        self._create_dynamic_parameter_fields() 

        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(pady=10, padx=10, fill="x")

        self.prev_button = ttk.Button(nav_frame, text="< Previous", command=lambda: self._nav_file(-1))
        self.prev_button.pack(side="left", padx=5)

        ttk.Label(nav_frame, text="Jump to file #:", anchor="e").pack(side="left", padx=(20, 5))
        vcmd_jump = (self.root.register(self._validate_jump_input), '%P')
        self.jump_entry = ttk.Entry(
            nav_frame,
            textvariable=self.jump_to_file_var,
            width=5,
            validate="focusout",
            validatecommand=vcmd_jump
        )
        self.jump_entry.pack(side="left", padx=5)
        self.jump_entry.bind("<Return>", self._jump_to_file)
        
        self.next_button = ttk.Button(nav_frame, text="Next >", command=lambda: self._nav_file(1))
        self.next_button.pack(side="right", padx=5)

        status_bar = ttk.Label(self.root, textvariable=self.status_message_var, anchor="w")
        status_bar.pack(side="bottom", fill="x", pady=2, padx=10)

    def _display_current_file(self):
        if not self.all_depth_map_files or self.current_file_index == -1:
            # If there are no files, or no file is selected, revert GUI to 'no files' state.
            self._update_gui_for_no_files("No depth map files to display.")
            return

        current_file_data = self.all_depth_map_files[self.current_file_index]
        basename = current_file_data["basename"]
        
        # Update StringVar widgets
        self.current_filename_var.set(basename)
        self.file_status_var.set(f"{self.current_file_index + 1} of {len(self.all_depth_map_files)}")
        self.jump_to_file_var.set(str(self.current_file_index + 1))
        
        # Update DoubleVar widgets, ensuring rounding for display consistency
        for key, config in self.PARAMETER_CONFIG.items():
            # Get the value from the internal structure
            val = current_file_data.get(f"current_{key}", config["default"])
            # Set the Tkinter variable, which triggers the entry display update via trace
            self.param_vars[key].set(val)
        
        # Update button states based on current file and paste status
        self._update_navigation_buttons()

    def _exit_app(self):
        self._save_gui_config()
        self._save_current_file_gui_values() # NEW: Ensure current file's GUI values are in internal data
        self._save_progress(bulk_save=True) # Save all accumulated progress on exit
        self.root.destroy()

    def _generate_sidecar_jsons(self):
        if not self.all_depth_map_files:
            messagebox.showwarning("No Files", "No depth map files loaded to generate JSONs for.")
            return
        
        # 1. Ensure current file's GUI values are saved to the internal data structure
        self._save_current_file_gui_values() 
        # 2. Save all accumulated progress to the progress file
        self._save_progress(bulk_save=True)


        output_count = 0
        errors = []

        for file_data in self.all_depth_map_files:
            json_data = {}
            
            # Dynamically populate the sidecar JSON data for ALL parameters
            for key, config in self.PARAMETER_CONFIG.items():
                value = file_data[f"current_{key}"]          
                json_data[config["sidecar_key"]] = value
            
            base_name_without_ext = os.path.splitext(file_data["full_path"])[0]
            json_filename = base_name_without_ext + ".fssidecar" # NEW extension
            
            try:
                with open(json_filename, 'w') as f:
                    json.dump(json_data, f, indent=4)
                output_count += 1
            except Exception as e:
                errors.append(f"Failed to write {json_filename}: {e}")
        
        if errors:
            messagebox.showerror("Errors Generating FSSIDECARs", "\n".join(errors))
        self.status_message_var.set(f"Generated {output_count} FSSIDECAR sidecar files. {len(errors)} errors.")
        messagebox.showinfo("FSSIDECAR Generation Complete", f"Successfully generated {output_count} FSSIDECAR files.")

    def _get_video_frame_count(self, file_path):
        try:
            clip = VideoFileClip(file_path)
            fps = clip.fps
            duration = clip.duration
            if fps is None or duration is None:
                fps = 24 
                if duration is None: return 0 
            
            frames = math.ceil(duration * fps)
            clip.close()
            return frames
        except Exception as e:
            self.status_message_var.set(f"Error getting frame count for {os.path.basename(file_path)}: {e}")
            return 0

    def _jump_to_file(self, event=None):
        if not self.all_depth_map_files:
            return

        try:
            target_index = int(self.jump_to_file_var.get()) - 1
            if 0 <= target_index < len(self.all_depth_map_files):
                self._save_current_file_gui_values() # Save current GUI values before jumping
                
                self.current_file_index = target_index
                self._display_current_file()
            else:
                self.status_message_var.set(f"File number out of range (1-{len(self.all_depth_map_files)}).")
        except ValueError:
            self.status_message_var.set("Invalid file number for jump.")

    def _load_depth_maps(self):
        folder = self.current_folder_var.get()
        self.all_depth_map_files = []
        self.current_file_index = -1
        self.progress_save_data = {}
        self.applied_paste_data = {} # Clear any old temporary paste data

        if not folder or not os.path.isdir(folder):
            self.progress_file_path = None
            # Fix 1: Call with self.
            self._update_gui_for_no_files("Please select a valid depth map folder.") 
            return

        self.progress_file_path = os.path.join(folder, "sidecaredit_progress.json")
        self._load_progress() # This loads existing progress from disk into self.progress_save_data

        video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
        found_files_paths = []
        for ext in video_extensions:
            found_files_paths.extend(glob.glob(os.path.join(folder, ext)))
        sorted_files_paths = sorted(found_files_paths)

        if not sorted_files_paths:
            # Fix 1: Call with self.
            self._update_gui_for_no_files("No depth map video files found in the selected folder.")
            return

        cumulative_frames = 0
        total_files = len(sorted_files_paths)
        
        self.status_message_var.set(f"Loading {total_files} depth map files...")

        # Initialize last known values for carry-forward logic
        last_param_vals = {}
        for key, config in self.PARAMETER_CONFIG.items():
            last_param_vals[key] = config["default"]

        for i, full_path in enumerate(sorted_files_paths):
            basename = os.path.basename(full_path)
            total_frames = self._get_video_frame_count(full_path)
            
            saved_data = self.progress_save_data.get(basename)
            
            # 1. Start with values from the previous file (carry-forward)
            file_params = last_param_vals.copy()

            if saved_data and isinstance(saved_data, dict):
                # 2. If saved data exists, override carry-forward values
                for key, config in self.PARAMETER_CONFIG.items():
                    # Use saved value, or fall back to config default if key is missing
                    file_params[key] = saved_data.get(f"current_{key}", config["default"]) 
                
            elif saved_data is not None and isinstance(saved_data, (float, int)): # Old format compatibility
                # If old format (just convergence value), reset all others to default
                file_params["convergence"] = float(saved_data)
                for key, config in self.PARAMETER_CONFIG.items():
                    if key != "convergence":
                        file_params[key] = config["default"]
            
            # Construct the file_info dict dynamically
            file_info = {
                "full_path": full_path,
                "basename": basename,
                "total_frames": total_frames,
                "timeline_start_frame": cumulative_frames,
                "timeline_end_frame": cumulative_frames + total_frames - 1,
            }
            for key, val in file_params.items():
                 file_info[f"current_{key}"] = val
            
            self.all_depth_map_files.append(file_info)
            cumulative_frames += total_frames
            
            # 4. Update last values for the next iteration
            last_param_vals = file_params.copy()

            self.status_message_var.set(f"Loading file {i+1}/{total_files}: {basename} ({total_frames} frames)...")
            self.root.update_idletasks()

        self.current_file_index = 0
        # Fix 2: _display_current_file() correctly gets values from self.all_depth_map_files
        self._display_current_file() 
        self.status_message_var.set(f"Loaded {len(self.all_depth_map_files)} depth map files. Total frames: {cumulative_frames}.")

    def _load_folder_from_recent(self, folder_path):
        self.current_folder_var.set(folder_path)
        self._add_to_recent_folders(folder_path)
        self._load_depth_maps()

    def _load_folder_from_entry(self, event=None):
        folder_path = self.current_folder_var.get()
        if not folder_path:
            self.status_message_var.set("Error: Folder path cannot be empty.")
            return

        if not os.path.isdir(folder_path):
            self.status_message_var.set(f"Error: Not a valid directory: {folder_path}")
            return

        self._add_to_recent_folders(folder_path)
        self._load_depth_maps()

    def _load_fusion_export_file(self):
        if not self.all_depth_map_files:
            messagebox.showwarning("No Files", "Please load a depth map folder first.")
            return

        file_path = filedialog.askopenfilename(
            defaultextension=".fsexport",
            filetypes=[("Fusion Export Files", "*.fsexport.txt;*.fsexport"), ("All Files", "*.*")]
        )
        
        if not file_path:
            self.status_message_var.set("Fusion export file loading cancelled.")
            return

        self.status_message_var.set(f"Loading and processing {os.path.basename(file_path)}...")
        
        try:
            with open(file_path, 'r') as f:
                export_data = json.load(f)
        except json.JSONDecodeError as e:
            messagebox.showerror("File Error", f"Failed to parse JSON in {os.path.basename(file_path)}: {e}")
            self.status_message_var.set("Error: JSON parsing failed.")
            return
        except Exception as e:
            messagebox.showerror("File Error", f"Failed to read {os.path.basename(file_path)}: {e}")
            self.status_message_var.set("Error: File read failed.")
            return

        # 1. Extract and sort marker data
        markers = export_data.get("markers", [])
        if not markers:
            messagebox.showwarning("Data Warning", "No 'markers' found in the export file.")
            self.status_message_var.set("Warning: No markers found.")
            return
        
        # Sort markers by frame number
        markers.sort(key=lambda m: m['frame'])

        # 2. Apply parameters to files using carry-forward logic
        applied_count = 0
        
        # Initialize last values based on the first file's current (or default) values
        last_param_vals = {}
        for key, config in self.PARAMETER_CONFIG.items():
             last_param_vals[key] = self.all_depth_map_files[0].get(f"current_{key}", config["default"])
        
        for file_data in self.all_depth_map_files:
            file_start_frame = file_data["timeline_start_frame"]
            
            # Find the most relevant marker
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
                
                for key, config in self.PARAMETER_CONFIG.items():
                    fusion_key = config["fusion_key"]
                    default_val = config["default"]
                    
                    if fusion_key in marker_values:
                        current_param_vals[key] = marker_values.get(fusion_key, default_val)
                        updated_from_marker = True

                if updated_from_marker:
                    applied_count += 1
            
            # Update the internal file data structure
            for key, val in current_param_vals.items():
                file_data[f"current_{key}"] = val
            
            # Update last values for carry-forward to the next file
            last_param_vals = current_param_vals.copy()

        # 3. Update progress_save_data and save to disk
        # We must call _save_current_file_gui_values() first to ensure the currently
        # displayed file's values (which might have been manually changed) are in all_depth_map_files
        self._save_current_file_gui_values()
        self._save_progress(bulk_save=True)
        
        # 4. Update the GUI to show the new values for the current file
        self._display_current_file()
        
        self.status_message_var.set(f"Loaded {applied_count} files' parameters from {os.path.basename(file_path)}. Changes saved.")
    
    def _load_gui_config(self):
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, 'r') as f:
                    config = json.load(f)
                    self.recent_folders_list = config.get("recent_depth_maps_folders", [])
                    self.recent_folders_list = list(dict.fromkeys(self.recent_folders_list))
                    self.recent_folders_list = [f for f in self.recent_folders_list if os.path.isdir(f)]
                    self._trim_recent_folders()
            except json.JSONDecodeError as e:
                print(f"Error reading GUI config (corrupted JSON?): {e}")
                self.recent_folders_list = []
            except Exception as e:
                print(f"Error loading GUI config: {e}")
                self.recent_folders_list = []

    def _load_progress(self):
        if self.progress_file_path is None:
            self.progress_save_data = {}
            return

        if os.path.exists(self.progress_file_path):
            try:
                with open(self.progress_file_path, 'r') as f:
                    loaded_data = json.load(f)
                    self.progress_save_data = {}
                    for key, value in loaded_data.items():
                        if isinstance(value, dict):
                            # Modern format: check for all possible keys and use defaults if missing
                            progress_entry = value
                            for param_key, config in self.PARAMETER_CONFIG.items():
                                progress_entry.setdefault(f"current_{param_key}", config["default"])
                            self.progress_save_data[key] = progress_entry
                        else:
                            # Old format compatibility (only convergence)
                            progress_entry = {}
                            for param_key, config in self.PARAMETER_CONFIG.items():
                                progress_entry[f"current_{param_key}"] = float(value) if param_key == "convergence" else config["default"]
                            self.progress_save_data[key] = progress_entry
                self.status_message_var.set(f"Loaded progress from {os.path.basename(self.progress_file_path)}")
            except json.JSONDecodeError as e:
                self.status_message_var.set(f"Error reading progress file (corrupted JSON?): {e}")
                self.progress_save_data = {}
            except Exception as e:
                self.status_message_var.set(f"Error loading progress: {e}")
                self.progress_save_data = {}
        else:
            self.progress_save_data = {}

    def _nav_file(self, direction):
        if not self.all_depth_map_files:
            return
        
        self._save_current_file_gui_values()
        
        new_index = self.current_file_index + direction
        if 0 <= new_index < len(self.all_depth_map_files):
            self.current_file_index = new_index
            self._display_current_file()
        else:
            self._update_navigation_buttons()
            self.status_message_var.set("Already at first/last file.")

    def _save_current_file_gui_values(self):
        """Saves the current GUI entry values to the internal all_depth_map_files list."""
        if self.current_file_index != -1 and self.all_depth_map_files:
            current_data = self.all_depth_map_files[self.current_file_index]
            
            # Dynamic saving of parameters
            for key in self.PARAMETER_CONFIG:
                current_data[f"current_{key}"] = self.param_vars[key].get()

    def _save_gui_config(self):
        config = {
            "recent_depth_maps_folders": self.recent_folders_list
        }
        try:
            with open(self.config_file_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving GUI config: {e}")

    def _save_progress(self, bulk_save=False):
        if self.progress_file_path is None:
            self.status_message_var.set("Cannot save progress: No depth map folder selected.")
            return

        # Ensure all internal data (from manual edits or paste) is correctly reflected in progress_save_data
        # before writing to disk.
        # This loop iterates through all files in 'all_depth_map_files' and updates/adds them to 'progress_save_data'.
        for file_data in self.all_depth_map_files:
            progress_entry = {}
            for key, config in self.PARAMETER_CONFIG.items():
                # Use the sidecar key for persistence in progress file
                progress_entry[f"current_{key}"] = file_data[f"current_{key}"]                
            self.progress_save_data[file_data["basename"]] = progress_entry
        
        try:
            with open(self.progress_file_path, 'w') as f:
                json.dump(self.progress_save_data, f, indent=4)
            if not bulk_save: # Only show explicit message if not part of a larger operation
                self.status_message_var.set(f"Progress for all files saved to {os.path.basename(self.progress_file_path)}")
            else:
                 self.status_message_var.set(f"Progress saved (bulk operation) to {os.path.basename(self.progress_file_path)}")
        except Exception as e:
            self.status_message_var.set(f"Error saving progress: {e}")

    def _trim_recent_folders(self):
        self.recent_folders_list = self.recent_folders_list[:self.MAX_RECENT_FOLDERS]

    def _update_entry_display(self, param_key, *args):
            try:
                val = self.param_vars[param_key].get()
                decimals = self.PARAMETER_CONFIG[param_key]["decimals"]
                
                entry = self.param_entry_widgets[param_key]
                entry.delete(0, tk.END)
                
                if self.PARAMETER_CONFIG[param_key]["type"] is float:
                    entry.insert(0, f"{val:.{decimals}f}")
                else:
                    entry.insert(0, str(int(val)))
            except _tkinter.TclError:
                pass
            except ValueError:
                pass

    def _update_gui_for_no_files(self, message):
        self.current_filename_var.set(message)
        self.file_status_var.set("0 of 0")
        # Round default values for consistent display
        for key, config in self.PARAMETER_CONFIG.items():
            self.param_vars[key].set(config["default"])
            self.param_entry_widgets[key].config(state="disabled") # Ensure all are disabled
            
        self.jump_to_file_var.set("1")
        self.status_message_var.set(message)
        self._update_navigation_buttons()

    def _update_navigation_buttons(self):
        total_files = len(self.all_depth_map_files)
        if total_files == 0:
            state = "disabled"
        else:
            state = "normal"

        self.prev_button.config(state="normal" if self.current_file_index > 0 else "disabled")
        self.next_button.config(state="normal" if self.current_file_index < total_files - 1 else "disabled")
        
        self.jump_entry.config(state=state)
        for key in self.PARAMETER_CONFIG:
            widget = self.param_entry_widgets.get(key)
            if widget:
                widget.config(state=state)
        
    def _update_recent_folders_menu(self):
        self.recent_folders_submenu.delete(0, tk.END)
        if not self.recent_folders_list:
            self.recent_folders_submenu.add_command(label="(No recent folders)", state="disabled")
            return

        for folder_path in self.recent_folders_list:
            self.recent_folders_submenu.add_command(label=folder_path, command=lambda p=folder_path: self._load_folder_from_recent(p))

    def _validate_dynamic_input(self, p, param_key, is_return=False):
        config = self.PARAMETER_CONFIG[param_key]
        var_type = config["type"]
        min_val = config["min"]
        max_val = config["max"]
        decimals = config["decimals"]
        
        # When triggering validation on Return, the value 'p' might be the current variable value
        if is_return:
            p = self.param_vars[param_key].get()
        
        # Tkinter validation for focusout passes the new value as a string 'p'
        if not is_return and p == "":
            self.status_message_var.set(f"Error: {config['label']} cannot be empty.")
            return False
            
        try:
            val = var_type(p)
            
            if min_val <= val <= max_val:
                self.status_message_var.set("Ready.")
                
                # Round value based on config
                rounded_val = round(val, decimals) 
                
                # Update Tkinter variable
                self.param_vars[param_key].set(rounded_val) 
                
                # Update internal data structure
                if self.current_file_index != -1 and self.all_depth_map_files:
                    self.all_depth_map_files[self.current_file_index][f"current_{param_key}"] = rounded_val
                
                return True
            else:
                self.status_message_var.set(f"Error: {config['label']} must be between {min_val} and {max_val}.")
                return False
        except ValueError:
            self.status_message_var.set(f"Error: Invalid {var_type.__name__} for {config['label']}.")
            return False
    
    def _validate_jump_input(self, p):
        if p == "":
            return True
        try:
            val = int(p)
            if 1 <= val <= len(self.all_depth_map_files):
                self.status_message_var.set("Ready.")
                return True
            else:
                self.status_message_var.set(f"Error: File number must be between 1 and {len(self.all_depth_map_files)}.")
                return False
        except ValueError:
            self.status_message_var.set("Error: Invalid number for file jump.")
            return False
    
if __name__ == "__main__":
    root = tk.Tk()
    app = SidecarEditTool(root)
    root.protocol("WM_DELETE_WINDOW", app._exit_app)
    root.mainloop()