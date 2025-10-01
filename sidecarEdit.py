import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import json
import glob
import math
import _tkinter
from moviepy.editor import VideoFileClip

class SidecarEditTool:
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
        self.convergence_var = tk.DoubleVar(value=0.5)
        self.max_disparity_var = tk.DoubleVar(value=35.0)
        self.jump_to_file_var = tk.StringVar(value="1")
        self.status_message_var = tk.StringVar(value="Ready.")
        
        self.frame_overlap_var = tk.IntVar(value=5)
        self.input_bias_var = tk.DoubleVar(value=0.8)
        self.inpaint_enabled_var = tk.BooleanVar(value=False)

        self.all_depth_map_files = []
        self.current_file_index = -1
        
        self.progress_save_data = {}

        self.config_file_path = "sidecaredit_config.json"
        self.progress_file_path = None

        # --- Step 2: Load Configuration (this doesn't interact with GUI directly yet) ---
        self._load_gui_config()

        # --- Step 3: Create all GUI widgets and menus ---
        self._create_menu()
        self._create_widgets() # <-- ALL WIDGETS ARE NOW GUARANTEED TO BE CREATED HERE

        # --- Step 4: Set up Traces (after vars and widgets are defined) ---
        self.convergence_var.trace_add("write", self._update_convergence_entry_display)
        self.max_disparity_var.trace_add("write", self._update_max_disparity_entry_display)
        self.inpaint_enabled_var.trace_add("write", self._update_inpaint_controls_state)
        
        # --- Step 5: Set Initial UI States (these methods now safely configure existing widgets) ---
        self._update_navigation_buttons()
        self._update_inpaint_controls_state() # Initialize the state of inpaint controls

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

        # NEW: Main container for Depth and Inpaint Parameters to sit side-by-side
        params_outer_frame = ttk.Frame(self.root)
        params_outer_frame.pack(pady=5, padx=10, fill="x", expand=True)

        # NEW: Combined Depth Parameters LabelFrame (left side)
        depth_params_frame = ttk.LabelFrame(params_outer_frame, text="Depth Parameters")
        depth_params_frame.pack(side="left", padx=(0, 5), fill="both", expand=True) # Fill both vertically and expand
        depth_params_frame.grid_columnconfigure(1, weight=1) # Allow parameter entry to expand

        # Max Disparity (now directly in depth_params_frame)
        ttk.Label(depth_params_frame, text="Max Disparity (0-100, float):").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        vcmd_disparity = (self.root.register(self._validate_max_disparity_input), '%P')
        self.max_disparity_entry = ttk.Entry(
            depth_params_frame, # Reparented to depth_params_frame
            textvariable=self.max_disparity_var,
            width=10,
            validate="focusout",
            validatecommand=vcmd_disparity
        )
        self.max_disparity_entry.grid(row=0, column=1, sticky="w", padx=5, pady=2) # Using grid
        self.max_disparity_entry.bind("<Return>", lambda e: self._validate_max_disparity_input(self.max_disparity_var.get()))

        # Convergence Plane (now directly in depth_params_frame)
        ttk.Label(depth_params_frame, text="Convergence Plane (0.0-1.0, float):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        vcmd = (self.root.register(self._validate_convergence_input), '%P')
        self.convergence_entry = ttk.Entry(
            depth_params_frame, # Reparented to depth_params_frame
            textvariable=self.convergence_var,
            width=10,
            validate="focusout",
            validatecommand=vcmd
        )
        self.convergence_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2) # Using grid
        self.convergence_entry.bind("<Return>", lambda e: self._validate_convergence_input(self.convergence_var.get()))

        # Existing Inpaint Parameters section (right side) - now also child of params_outer_frame
        inpaint_frame = ttk.LabelFrame(params_outer_frame, text="Inpaint Parameters (Optional)") # Reparented
        inpaint_frame.pack(side="left", padx=(5, 0), fill="both", expand=True) # Fill both vertically and expand

        self.enable_inpaint_check = ttk.Checkbutton(
            inpaint_frame,
            text="Enable Inpaint Parameters for this video",
            variable=self.inpaint_enabled_var,
            command=self._update_inpaint_controls_state # Call on click
        )
        self.enable_inpaint_check.grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=2)
        inpaint_frame.grid_columnconfigure(1, weight=1) # Allow parameter entry to expand

        ttk.Label(inpaint_frame, text="Frame Overlap (integer, >=0):").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        vcmd_overlap = (self.root.register(self._validate_frame_overlap_input), '%P')
        self.frame_overlap_entry = ttk.Entry(
            inpaint_frame,
            textvariable=self.frame_overlap_var,
            width=10,
            validate="focusout",
            validatecommand=vcmd_overlap
        )
        self.frame_overlap_entry.grid(row=1, column=1, sticky="w", padx=5, pady=2)
        self.frame_overlap_entry.bind("<Return>", self._validate_frame_overlap_input_wrapper)

        ttk.Label(inpaint_frame, text="Input Bias (0.0-1.0, float):").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        vcmd_bias = (self.root.register(self._validate_input_bias_input), '%P')
        self.input_bias_entry = ttk.Entry(
            inpaint_frame,
            textvariable=self.input_bias_var,
            width=10,
            validate="focusout",
            validatecommand=vcmd_bias
        )
        self.input_bias_entry.grid(row=2, column=1, sticky="w", padx=5, pady=2)
        self.input_bias_entry.bind("<Return>", self._validate_input_bias_input_wrapper)
        # END Inpaint Parameters Section

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
        self.convergence_var.set(round(current_file_data["current_convergence"], 2))
        self.max_disparity_var.set(round(current_file_data["current_max_disparity"], 1))
        self.frame_overlap_var.set(current_file_data.get("current_frame_overlap", 5))
        self.input_bias_var.set(round(current_file_data.get("current_input_bias", 0.8), 2))
        self.inpaint_enabled_var.set(current_file_data.get("inpaint_enabled", False))
        self._update_inpaint_controls_state()

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
        
        if self.current_file_index != -1 and self.all_depth_map_files:
            current_data = self.all_depth_map_files[self.current_file_index]
            self.progress_save_data[current_data["basename"]] = {
                "convergence_plane": self.convergence_var.get(),
                "max_disparity": self.max_disparity_var.get()
            }
            self._save_progress(bulk_save=True)

        output_count = 0
        errors = []

        for file_data in self.all_depth_map_files:
            basename = file_data["basename"]
            
            convergence_value = file_data["current_convergence"]
            max_disparity_value = file_data["current_max_disparity"]

            json_data = {
                "convergence_plane": convergence_value,
                "max_disparity": max_disparity_value
            }

            if file_data.get("inpaint_enabled", False):
                json_data["frame_overlap"] = file_data.get("current_frame_overlap", 5)
                json_data["input_bias"] = file_data.get("current_input_bias", 0.8)
            
            base_name_without_ext = os.path.splitext(file_data["full_path"])[0]
            json_filename = base_name_without_ext + ".fssidecar"
            
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
                self._save_current_file_gui_values() # NEW: Save current GUI values before jumping
                
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
        last_conv_val = 0.5
        last_disp_val = 35.0
        last_overlap_val = 5
        last_bias_val = 0.8
        last_inpaint_enabled = False # Default disabled

        for i, full_path in enumerate(sorted_files_paths):
            basename = os.path.basename(full_path)
            total_frames = self._get_video_frame_count(full_path)
            
            saved_data = self.progress_save_data.get(basename)
            
            conv_val = last_conv_val
            disp_val = last_disp_val
            overlap_val = last_overlap_val
            bias_val = last_bias_val
            inpaint_enabled_val = last_inpaint_enabled
            
            if saved_data:
                # If saved data exists, use it
                if isinstance(saved_data, dict):
                    conv_val = saved_data.get("convergence_plane", 0.5)
                    disp_val = saved_data.get("max_disparity", 35.0)
                    overlap_val = saved_data.get("frame_overlap", 5)
                    bias_val = saved_data.get("input_bias", 0.8)
                    inpaint_enabled_val = saved_data.get("inpaint_enabled", False)
                    # ADD END
                elif isinstance(saved_data, (float, int)): # Old format compatibility
                    conv_val = float(saved_data)
                    disp_val = 35.0 # Default for old format
                    overlap_val = 5
                    bias_val = 0.8
                    inpaint_enabled_val = False
            else:
                # If NO saved data, use the last determined values (carry-forward)
                conv_val = last_conv_val
                disp_val = last_disp_val
                overlap_val = last_overlap_val
                bias_val = last_bias_val
                inpaint_enabled_val = last_inpaint_enabled
            
            file_info = {
                "full_path": full_path,
                "basename": basename,
                "total_frames": total_frames,
                "timeline_start_frame": cumulative_frames,
                "timeline_end_frame": cumulative_frames + total_frames - 1,
                "current_convergence": conv_val,
                "current_max_disparity": disp_val,
                "current_frame_overlap": overlap_val,
                "current_input_bias": bias_val,
                "inpaint_enabled": inpaint_enabled_val
            }
            self.all_depth_map_files.append(file_info)
            cumulative_frames += total_frames
            
            # Update last_conv_val and last_disp_val for the next iteration
            last_conv_val = conv_val
            last_disp_val = disp_val
            last_overlap_val = overlap_val
            last_bias_val = bias_val
            last_inpaint_enabled = inpaint_enabled_val

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
        last_conv = self.all_depth_map_files[0]["current_convergence"]
        last_disp = self.all_depth_map_files[0]["current_max_disparity"]
        
        for file_data in self.all_depth_map_files:
            file_start_frame = file_data["timeline_start_frame"]
            
            # Find the most relevant marker (frame <= file_start_frame)
            relevant_marker = None
            for marker in markers:
                if marker['frame'] <= file_start_frame:
                    relevant_marker = marker
                else:
                    break
            
            conv_val = last_conv
            disp_val = last_disp

            if relevant_marker and relevant_marker.get('values'):
                # Use values from the most relevant marker
                conv_val = relevant_marker['values'].get('Convergence', last_conv)
                disp_val = relevant_marker['values'].get('MaxDisparity', last_disp)
                applied_count += 1
            
            # Update the internal file data structure
            file_data["current_convergence"] = conv_val
            file_data["current_max_disparity"] = disp_val
            
            # Update last values for carry-forward to the next file
            last_conv = conv_val
            last_disp = disp_val

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
                            self.progress_save_data[key] = value
                            self.progress_save_data[key].setdefault("frame_overlap", 5)
                            self.progress_save_data[key].setdefault("input_bias", 0.8)
                            self.progress_save_data[key].setdefault("inpaint_enabled", False)
                        else:
                            self.progress_save_data[key] = {
                                "convergence_plane": float(value),
                                "max_disparity": 35.0,
                                "frame_overlap": 5,
                                "input_bias": 0.8,
                                "inpaint_enabled": False
                            }
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
        """Saves the current GUI slider/entry values to the internal all_depth_map_files list."""
        if self.current_file_index != -1 and self.all_depth_map_files:
            current_data = self.all_depth_map_files[self.current_file_index]
            # Ensure the values are correctly read from the DoubleVars, not just what was last set
            current_data["current_convergence"] = self.convergence_var.get()
            current_data["current_max_disparity"] = self.max_disparity_var.get()
            current_data["current_frame_overlap"] = self.frame_overlap_var.get()
            current_data["current_input_bias"] = self.input_bias_var.get()
            current_data["inpaint_enabled"] = self.inpaint_enabled_var.get()
            # print(f"DEBUG: Saved GUI values for {current_data['basename']} (Conv: {current_data['current_convergence']:.4f}, Disp: {current_data['current_max_disparity']:.2f})")

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
            self.progress_save_data[file_data["basename"]] = {
                "convergence_plane": file_data["current_convergence"],
                "max_disparity": file_data["current_max_disparity"],
                "frame_overlap": file_data["current_frame_overlap"],
                "input_bias": file_data["current_input_bias"],
                "inpaint_enabled": file_data["inpaint_enabled"]
            }
        
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

    def _update_convergence_entry_display(self, *args):
        try:
            val = self.convergence_var.get()
            self.convergence_entry.delete(0, tk.END)
            self.convergence_entry.insert(0, f"{val:.2f}") # Format to 2 decimal places
        except _tkinter.TclError:
            pass # Ignore TclErrors during partial updates
        except ValueError:
            pass

    def _update_gui_for_no_files(self, message):
        self.current_filename_var.set(message)
        self.file_status_var.set("0 of 0")
        # Round default values for consistent display
        self.convergence_var.set(round(0.5, 2))
        self.max_disparity_var.set(round(35.0, 1))
        self.jump_to_file_var.set("1")
        self.status_message_var.set(message)
        self._update_navigation_buttons()

    def _update_inpaint_controls_state(self, *args):
        # Determine the state based on whether files are loaded AND the checkbox state
        if not self.all_depth_map_files or self.current_file_index == -1:
            # If no files are loaded, disable everything
            entry_state = "disabled"
            checkbox_state = "disabled"
        else:
            # Files are loaded, checkbox can always be toggled
            checkbox_state = "normal"
            if self.inpaint_enabled_var.get():
                entry_state = "normal"
            else:
                entry_state = "disabled"

        self.frame_overlap_entry.config(state=entry_state)
        self.input_bias_entry.config(state=entry_state)
        self.enable_inpaint_check.config(state=checkbox_state)
    
    def _update_max_disparity_entry_display(self, *args):
        try:
            val = self.max_disparity_var.get()
            self.max_disparity_entry.delete(0, tk.END)
            self.max_disparity_entry.insert(0, f"{val:.1f}") # Format to 1 decimal place
        except _tkinter.TclError:
            pass # Ignore TclErrors during partial updates
        except ValueError:
            pass

    def _update_navigation_buttons(self):
        total_files = len(self.all_depth_map_files)
        if total_files == 0:
            state = "disabled"
        else:
            state = "normal"

        self.prev_button.config(state="normal" if self.current_file_index > 0 else "disabled")
        self.next_button.config(state="normal" if self.current_file_index < total_files - 1 else "disabled")
        
        self.jump_entry.config(state=state)
        self.convergence_entry.config(state=state)
        self.max_disparity_entry.config(state=state)
        self.frame_overlap_entry.config(state=state)
        self.input_bias_entry.config(state=state)
        self.enable_inpaint_check.config(state=state) # Temporarily set based on file presence
        
        # This call will then apply the specific inpaint enable/disable logic based on self.inpaint_enabled_var
        self._update_inpaint_controls_state() 
        
    def _update_recent_folders_menu(self):
        self.recent_folders_submenu.delete(0, tk.END)
        if not self.recent_folders_list:
            self.recent_folders_submenu.add_command(label="(No recent folders)", state="disabled")
            return

        for folder_path in self.recent_folders_list:
            self.recent_folders_submenu.add_command(label=folder_path, command=lambda p=folder_path: self._load_folder_from_recent(p))

    def _validate_convergence_input(self, p):
        if p == "":
            return False
        try:
            val = float(p)
            if 0.0 <= val <= 1.0:
                self.status_message_var.set("Ready.")
                rounded_val = round(val, 2) # NEW: Round here
                self.convergence_var.set(rounded_val) 
                if self.current_file_index != -1 and self.all_depth_map_files:
                    self.all_depth_map_files[self.current_file_index]["current_convergence"] = rounded_val # NEW: Store rounded
                return True
            else:
                self.status_message_var.set("Error: Convergence must be between 0.0 and 1.0.")
                return False
        except ValueError:
            self.status_message_var.set("Error: Invalid number for convergence.")
            return False

    def _validate_max_disparity_input(self, p):
        if p == "":
            self.status_message_var.set("Error: Max Disparity cannot be empty.")
            return False
        try:
            val = float(p)
            if 0.0 <= val <= 100.0:
                self.status_message_var.set("Ready.")
                rounded_val = round(val, 1) # NEW: Round here
                self.max_disparity_var.set(rounded_val)
                if self.current_file_index != -1 and self.all_depth_map_files:
                    self.all_depth_map_files[self.current_file_index]["current_max_disparity"] = rounded_val # NEW: Store rounded
                return True
            else:
                self.status_message_var.set("Error: Max Disparity must be between 0.0 and 100.0.")
                return False
        except ValueError:
            self.status_message_var.set("Error: Invalid number for max disparity.")
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

    def _validate_frame_overlap_input_wrapper(self, event=None):
        # Called when <Return> is pressed in the entry.
        # It triggers the focusout validation implicitly if the content changed,
        # but this ensures explicit validation on Enter.
        self._validate_frame_overlap_input(self.frame_overlap_var.get())

    def _validate_input_bias_input_wrapper(self, event=None):
        # Called when <Return> is pressed in the entry.
        self._validate_input_bias_input(self.input_bias_var.get())
    
    def _validate_frame_overlap_input(self, p):
        if not self.inpaint_enabled_var.get():
            # If inpaint is not enabled, we don't strictly validate the input field
            # as its value won't be used/saved.
            self.status_message_var.set("Ready (Inpaint disabled).")
            return True
        if p == "":
            self.status_message_var.set("Error: Frame Overlap cannot be empty.")
            return False
        try:
            val = int(p)
            if val >= 0:
                self.status_message_var.set("Ready.")
                self.frame_overlap_var.set(val)
                if self.current_file_index != -1 and self.all_depth_map_files:
                    self.all_depth_map_files[self.current_file_index]["current_frame_overlap"] = val
                return True
            else:
                self.status_message_var.set("Error: Frame Overlap must be 0 or greater.")
                return False
        except ValueError:
            self.status_message_var.set("Error: Invalid integer for Frame Overlap.")
            return False

    def _validate_input_bias_input(self, p):
        if not self.inpaint_enabled_var.get():
            # If inpaint is not enabled, we don't strictly validate the input field
            # as its value won't be used/saved.
            self.status_message_var.set("Ready (Inpaint disabled).")
            return True
        if p == "":
            self.status_message_var.set("Error: Input Bias cannot be empty.")
            return False
        try:
            val = float(p)
            if 0.0 <= val <= 1.0:
                self.status_message_var.set("Ready.")
                rounded_val = round(val, 2) # Often 2 decimal places for bias
                self.input_bias_var.set(rounded_val)
                if self.current_file_index != -1 and self.all_depth_map_files:
                    self.all_depth_map_files[self.current_file_index]["current_input_bias"] = rounded_val
                return True
            else:
                self.status_message_var.set("Error: Input Bias must be between 0.0 and 1.0.")
                return False
        except ValueError:
            self.status_message_var.set("Error: Invalid number for Input Bias.")
            return False
    
if __name__ == "__main__":
    root = tk.Tk()
    app = SidecarEditTool(root)
    root.protocol("WM_DELETE_WINDOW", app._exit_app)
    root.mainloop()