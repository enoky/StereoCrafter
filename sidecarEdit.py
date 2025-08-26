import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import os
import json
import glob
import math
import _tkinter # Import _tkinter for error handling

class ConvergenceJSONTool:
    def __init__(self, root):
        self.root = root
        self.root.title("SidecarEdit JSON Sidecar Creator") # Changed title
        self.root.geometry("600x500") # Initial window size

        # --- Variables ---
        self.current_folder_var = tk.StringVar(value="") # Holds the currently selected folder
        self.recent_folders_list = [] # Stores list of recent folder paths
        self.MAX_RECENT_FOLDERS = 10 # Max number of recent folders to keep

        self.current_filename_var = tk.StringVar(value="No file loaded")
        self.file_status_var = tk.StringVar(value="0 of 0")
        self.convergence_var = tk.DoubleVar(value=0.5) # Slider needs DoubleVar
        self.max_disparity_var = tk.DoubleVar(value=35.0) # NEW: Slider for max disparity (default 35)
        self.jump_to_file_var = tk.StringVar(value="1")
        self.status_message_var = tk.StringVar(value="Ready.")

        self.all_depth_map_files = [] # Stores full paths to depth maps
        self.current_file_index = -1  # -1 means no file loaded
        # self.progress_save_data will now store {filename_basename: {"convergence_plane": value, "max_disparity": value}}
        self.progress_save_data = {}  

        self.config_file_path = "sidecaredit_config.json" # Changed config filename
        self.progress_file_path = None  # Will be set dynamically based on folder

        # --- Load Configuration and Progress on startup ---
        self._load_gui_config()
        # Progress will be loaded dynamically when a folder is selected

        # --- GUI Layout ---
        self._create_menu() # Create the File menu first
        self._create_widgets()
        self._update_navigation_buttons() # Initial state of buttons

        # Sync slider and entry
        self.convergence_var.trace_add("write", self._sync_convergence_entry_from_slider)
        self.max_disparity_var.trace_add("write", self._sync_max_disparity_entry_from_slider) # NEW: Sync disparity slider and entry
        
        # Initial folder load if any in recent list
        if self.recent_folders_list:
            # Set the current folder var to the most recent one
            self.current_folder_var.set(self.recent_folders_list[0])
            self._load_depth_maps()

    def _create_menu(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

        self.file_menu.add_command(label="Load Depth Map Folder...", command=self._browse_folder)
        
        # Recent Folders Submenu
        self.recent_folders_submenu = tk.Menu(self.file_menu, tearoff=0)
        self.file_menu.add_cascade(label="Recent Folders", menu=self.recent_folders_submenu)
        self._update_recent_folders_menu() # Populate initial list

        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save Progress", command=self._save_progress)
        self.file_menu.add_command(label="Generate Sidecar JSONs...", command=self._generate_sidecar_jsons)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self._exit_app)

    def _create_widgets(self):
        # Folder Selection Frame
        folder_frame = ttk.LabelFrame(self.root, text="Depth Map Folder")
        folder_frame.pack(pady=10, padx=10, fill="x")
        
        # Entry for folder path (allows pasting)
        self.folder_entry = ttk.Entry(folder_frame, textvariable=self.current_folder_var, width=60)
        self.folder_entry.grid(row=0, column=0, padx=5, pady=5)
        self.folder_entry.bind("<Return>", self._load_folder_from_entry) # Load on Enter key
        self.folder_entry.bind("<FocusOut>", self._load_folder_from_entry) # Load on focus out

        ttk.Button(folder_frame, text="Browse", command=self._browse_folder).grid(row=0, column=1, padx=5, pady=5)

        # Current File Info Frame
        file_info_frame = ttk.LabelFrame(self.root, text="Current Depth Map")
        file_info_frame.pack(pady=5, padx=10, fill="x")

        ttk.Label(file_info_frame, text="Filename:").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        ttk.Label(file_info_frame, textvariable=self.current_filename_var, wraplength=400).grid(row=0, column=1, sticky="w", padx=5, pady=2)

        # NEW: Max Disparity Control Frame (placed above Convergence)
        max_disparity_frame = ttk.LabelFrame(self.root, text="Max Disparity (0-100)")
        max_disparity_frame.pack(pady=5, padx=10, fill="x")

        # Slider for Max Disparity
        self.max_disparity_slider = ttk.Scale(
            max_disparity_frame,
            from_=0.0,
            to=100.0,
            orient=tk.HORIZONTAL,
            variable=self.max_disparity_var
        )
        self.max_disparity_slider.pack(fill="x", padx=5, pady=5)

        # Entry for numerical input for Max Disparity, synced with slider
        vcmd_disparity = (self.root.register(self._validate_max_disparity_input), '%P')
        self.max_disparity_entry = ttk.Entry(
            max_disparity_frame,
            textvariable=self.max_disparity_var,
            width=10,
            validate="focusout", # Validate on losing focus
            validatecommand=vcmd_disparity
        )
        self.max_disparity_entry.pack(padx=5, pady=2)
        self.max_disparity_entry.bind("<Return>", self._sync_max_disparity_slider_from_entry) # Sync on Enter key

        # Convergence Control Frame (EXISTING)
        convergence_frame = ttk.LabelFrame(self.root, text="Convergence Plane (0.0=Nearest, 1.0=Furthest)")
        convergence_frame.pack(pady=5, padx=10, fill="x")

        # Slider
        self.convergence_slider = ttk.Scale(
            convergence_frame,
            from_=0.0,
            to=1.0,
            orient=tk.HORIZONTAL,
            variable=self.convergence_var
        )
        self.convergence_slider.pack(fill="x", padx=5, pady=5)

        # Entry for numerical input, synced with slider
        vcmd = (self.root.register(self._validate_convergence_input), '%P')
        self.convergence_entry = ttk.Entry(
            convergence_frame,
            textvariable=self.convergence_var,
            width=10,
            validate="focusout", # Validate on losing focus
            validatecommand=vcmd
        )
        self.convergence_entry.pack(padx=5, pady=2)
        self.convergence_entry.bind("<Return>", self._sync_convergence_slider_from_entry) # Sync on Enter key

        # Navigation Frame
        nav_frame = ttk.Frame(self.root)
        nav_frame.pack(pady=10, padx=10, fill="x")

        self.prev_button = ttk.Button(nav_frame, text="< Previous", command=lambda: self._nav_file(-1))
        self.prev_button.pack(side="left", padx=5)

        ttk.Label(nav_frame, textvariable=self.file_status_var).pack(side="left", padx=10)

        self.next_button = ttk.Button(nav_frame, text="Next >", command=lambda: self._nav_file(1))
        self.next_button.pack(side="left", padx=5)

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

        # Status Bar
        status_bar = ttk.Label(self.root, textvariable=self.status_message_var, anchor="w")
        status_bar.pack(side="bottom", fill="x", pady=2, padx=10)

        # Action Buttons Frame (buttons removed as they are in the menu)
        action_frame = ttk.Frame(self.root)
        action_frame.pack(pady=10, padx=10, fill="x")

    def _update_recent_folders_menu(self):
        self.recent_folders_submenu.delete(0, tk.END) # Clear existing items
        if not self.recent_folders_list:
            self.recent_folders_submenu.add_command(label="(No recent folders)", state="disabled")
            return

        for folder_path in self.recent_folders_list:
            # Use lambda to capture folder_path for each command
            self.recent_folders_submenu.add_command(label=folder_path, command=lambda p=folder_path: self._load_folder_from_recent(p))

    def _load_folder_from_recent(self, folder_path):
        # Update current folder variable
        self.current_folder_var.set(folder_path)
        # Move selected folder to top of recent list
        self._add_to_recent_folders(folder_path) # This also updates the menu and saves config
        self._load_depth_maps()

    def _load_folder_from_entry(self, event=None):
        folder_path = self.current_folder_var.get()
        if not folder_path:
            self.status_message_var.set("Error: Folder path cannot be empty.")
            return

        if not os.path.isdir(folder_path):
            self.status_message_var.set(f"Error: Not a valid directory: {folder_path}")
            return

        self._add_to_recent_folders(folder_path) # Update recent list, menu, and save config
        self._load_depth_maps()

    def _browse_folder(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.current_folder_var.set(folder_selected)
            self._add_to_recent_folders(folder_selected) # Update recent list, menu, and save config
            self._load_depth_maps()

    def _add_to_recent_folders(self, folder_path):
        # Ensure it's in the list and at the top
        if folder_path in self.recent_folders_list:
            self.recent_folders_list.remove(folder_path)
        self.recent_folders_list.insert(0, folder_path)
        self._trim_recent_folders()
        self._update_recent_folders_menu() # Update the menu's list
        self._save_gui_config() # Save config with new recent list

    def _trim_recent_folders(self):
        self.recent_folders_list = self.recent_folders_list[:self.MAX_RECENT_FOLDERS]

    def _load_depth_maps(self):
        folder = self.current_folder_var.get()
        if not folder or not os.path.isdir(folder):
            self.all_depth_map_files = []
            self.current_file_index = -1
            self.progress_file_path = None # Reset if folder is invalid
            self._update_gui_for_no_files("Please select a valid depth map folder.")
            return

        # --- Change 1: Update progress file name ---
        self.progress_file_path = os.path.join(folder, "sidecaredit_progress.json")
        self._load_progress() # Load progress specifically for this new folder

        video_extensions = ('*.mp4', '*.avi', '*.mov', '*.mkv')
        found_files = []
        for ext in video_extensions:
            found_files.extend(glob.glob(os.path.join(folder, ext)))
        self.all_depth_map_files = sorted(found_files)

        if not self.all_depth_map_files:
            self.current_file_index = -1
            self._update_gui_for_no_files("No depth map video files found in the selected folder.")
            return

        self.current_file_index = 0
        self._display_current_file()
        self.status_message_var.set(f"Loaded {len(self.all_depth_map_files)} depth map files.")

    def _update_gui_for_no_files(self, message):
        self.current_filename_var.set(message)
        self.file_status_var.set("0 of 0")
        self.convergence_var.set(0.5)
        self.max_disparity_var.set(35.0) # NEW: Reset max disparity
        self.jump_to_file_var.set("1")
        self.status_message_var.set(message)
        self._update_navigation_buttons()

    def _display_current_file(self):
        if not self.all_depth_map_files or self.current_file_index == -1:
            self._update_gui_for_no_files("No depth map files to display.")
            return

        full_path = self.all_depth_map_files[self.current_file_index]
        filename = os.path.basename(full_path)
        self.current_filename_var.set(filename)
        self.file_status_var.set(f"{self.current_file_index + 1} of {len(self.all_depth_map_files)}")
        self.jump_to_file_var.set(str(self.current_file_index + 1))

        # --- Change 2: Lookup saved data using the basename ---
        saved_data = self.progress_save_data.get(filename)

        convergence_value = 0.5
        max_disparity_value = 35.0

        if saved_data is not None:
            if isinstance(saved_data, dict): # New format: dictionary of values
                convergence_value = saved_data.get("convergence_plane", 0.5)
                max_disparity_value = saved_data.get("max_disparity", 35.0)
            else: # Old format (before max_disparity): just a float (assumed to be convergence_plane)
                convergence_value = saved_data 
                # max_disparity_value remains default 35.0

        # Set the DoubleVars. This automatically updates the sliders and triggers the entry sync.
        self.convergence_var.set(convergence_value) 
        self.max_disparity_var.set(max_disparity_value) # NEW: Set max disparity value

        self._update_navigation_buttons()

    def _update_navigation_buttons(self):
        total_files = len(self.all_depth_map_files)
        if total_files == 0:
            self.prev_button.config(state="disabled")
            self.next_button.config(state="disabled")
            self.jump_entry.config(state="disabled")
            self.convergence_slider.config(state="disabled")
            self.convergence_entry.config(state="disabled")
            self.max_disparity_slider.config(state="disabled") # NEW
            self.max_disparity_entry.config(state="disabled")   # NEW
            return
        
        self.prev_button.config(state="normal" if self.current_file_index > 0 else "disabled")
        self.next_button.config(state="normal" if self.current_file_index < total_files - 1 else "disabled")
        self.jump_entry.config(state="normal")
        self.convergence_slider.config(state="normal")
        self.convergence_entry.config(state="normal")
        self.max_disparity_slider.config(state="normal") # NEW
        self.max_disparity_entry.config(state="normal")   # NEW

    def _sync_convergence_entry_from_slider(self, *args):
        try:
            val = self.convergence_var.get()
            self.convergence_entry.delete(0, tk.END)
            self.convergence_entry.insert(0, f"{val:.2f}")
        except _tkinter.TclError:
            pass
        except ValueError:
            pass

    def _sync_convergence_slider_from_entry(self, event=None):
        try:
            val = float(self.convergence_entry.get())
            self.convergence_var.set(val)
        except ValueError:
            pass

    # NEW: Sync methods for Max Disparity
    def _sync_max_disparity_entry_from_slider(self, *args):
        try:
            val = self.max_disparity_var.get()
            self.max_disparity_entry.delete(0, tk.END)
            self.max_disparity_entry.insert(0, f"{val:.2f}")
        except _tkinter.TclError:
            pass
        except ValueError:
            pass

    def _sync_max_disparity_slider_from_entry(self, event=None):
        try:
            val = float(self.max_disparity_entry.get())
            self.max_disparity_var.set(val)
        except ValueError:
            pass

    def _validate_convergence_input(self, p):
        if p == "":
            self.status_message_var.set("Error: Convergence cannot be empty.")
            return False
        try:
            val = float(p)
            if 0.0 <= val <= 1.0:
                self.status_message_var.set("Ready.")
                return True
            else:
                self.status_message_var.set("Error: Convergence must be between 0.0 and 1.0.")
                return False
        except ValueError:
            self.status_message_var.set("Error: Invalid number for convergence.")
            return False

    # NEW: Validation method for Max Disparity
    def _validate_max_disparity_input(self, p):
        if p == "":
            self.status_message_var.set("Error: Max Disparity cannot be empty.")
            return False
        try:
            val = float(p)
            if 0.0 <= val <= 100.0:
                self.status_message_var.set("Ready.")
                return True
            else:
                self.status_message_var.set("Error: Max Disparity must be between 0.0 and 100.0.")
                return False
        except ValueError:
            self.status_message_var.set("Error: Invalid number for max disparity.")
            return False

    def _validate_jump_input(self, p):
        if p == "":
            return True # Allow empty input temporarily
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

    def _nav_file(self, direction):
        if not self.all_depth_map_files:
            return

        self._save_progress() # Save current file's value before navigating

        new_index = self.current_file_index + direction
        if 0 <= new_index < len(self.all_depth_map_files):
            self.current_file_index = new_index
            self._display_current_file()
        else:
            self._update_navigation_buttons() # Re-disable if at ends
            self.status_message_var.set("Already at first/last file.")

    def _jump_to_file(self, event=None):
        if not self.all_depth_map_files:
            return

        try:
            target_index = int(self.jump_to_file_var.get()) - 1 # Convert to 0-based index
            if 0 <= target_index < len(self.all_depth_map_files):
                self._save_progress() # Save current file's convergence before jumping
                
                self.current_file_index = target_index
                self._display_current_file()
            else:
                self.status_message_var.set(f"File number out of range (1-{len(self.all_depth_map_files)}).")
        except ValueError:
            self.status_message_var.set("Invalid file number for jump.")

    def _save_progress(self):
        if self.progress_file_path is None:
            self.status_message_var.set("Cannot save progress: No depth map folder selected.")
            return

        if self.current_file_index != -1 and self.all_depth_map_files:
            current_file_path = self.all_depth_map_files[self.current_file_index]
            # --- Change 3: Store basename as the key ---
            current_file_basename = os.path.basename(current_file_path)
            self.progress_save_data[current_file_basename] = {
                "convergence_plane": self.convergence_var.get(),
                "max_disparity": self.max_disparity_var.get()
            }
        
        try:
            with open(self.progress_file_path, 'w') as f:
                json.dump(self.progress_save_data, f, indent=4)
            self.status_message_var.set(f"Progress saved to {os.path.basename(self.progress_file_path)}")
        except Exception as e:
            self.status_message_var.set(f"Error saving progress: {e}")

    def _load_progress(self):
        if self.progress_file_path is None:
            self.progress_save_data = {} # Ensure it's clear if no folder selected
            return

        if os.path.exists(self.progress_file_path):
            try:
                with open(self.progress_file_path, 'r') as f:
                    self.progress_save_data = json.load(f)
                self.status_message_var.set(f"Loaded progress from {os.path.basename(self.progress_file_path)}")
            except json.JSONDecodeError as e:
                self.status_message_var.set(f"Error reading progress file (corrupted JSON?): {e}")
                self.progress_save_data = {}
            except Exception as e:
                self.status_message_var.set(f"Error loading progress: {e}")
                self.progress_save_data = {}
        else:
            self.progress_save_data = {} # Clear progress if file doesn't exist for new folder

    def _generate_sidecar_jsons(self):
        if not self.all_depth_map_files:
            messagebox.showwarning("No Files", "No depth map files loaded to generate JSONs for.")
            return
        
        # Ensure all current edits are saved before generating
        self._save_progress() 

        output_count = 0
        errors = []

        for full_path in self.all_depth_map_files:
            # --- Change 4: Lookup saved data using the basename ---
            filename_only = os.path.basename(full_path) 
            saved_data = self.progress_save_data.get(filename_only)
            
            convergence_value = 0.5
            max_disparity_value = 35.0

            if saved_data is not None:
                if isinstance(saved_data, dict):
                    convergence_value = saved_data.get("convergence_plane", 0.5)
                    max_disparity_value = saved_data.get("max_disparity", 35.0)
                else: # Old format, only convergence was saved
                    convergence_value = saved_data

            json_data = {
                "convergence_plane": convergence_value,
                "max_disparity": max_disparity_value # NEW: Include max_disparity
            }
            
            # Construct sidecar JSON filename (e.g., video.mp4 -> video.mp4.json)
            base_name_without_ext = os.path.splitext(full_path)[0] # Get the path without the last extension
            json_filename = base_name_without_ext + ".json"
            
            try:
                with open(json_filename, 'w') as f:
                    json.dump(json_data, f, indent=4)
                output_count += 1
            except Exception as e:
                errors.append(f"Failed to write {json_filename}: {e}")
        
        if errors:
            messagebox.showerror("Errors Generating JSONs", "\n".join(errors))
        self.status_message_var.set(f"Generated {output_count} JSON sidecar files. {len(errors)} errors.")
        messagebox.showinfo("JSON Generation Complete", f"Successfully generated {output_count} JSON files.")

    def _save_gui_config(self):
        config = {
            "recent_depth_maps_folders": self.recent_folders_list
        }
        try:
            with open(self.config_file_path, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving GUI config: {e}")

    def _load_gui_config(self):
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, 'r') as f:
                    config = json.load(f)
                    # Load recent folders, ensuring it's a list
                    self.recent_folders_list = config.get("recent_depth_maps_folders", [])
                    # Clean up list, remove duplicates and invalid paths (optional but good)
                    self.recent_folders_list = list(dict.fromkeys(self.recent_folders_list)) # Remove duplicates
                    self.recent_folders_list = [f for f in self.recent_folders_list if os.path.isdir(f)] # Only keep valid paths
                    self._trim_recent_folders() # Ensure it's trimmed to MAX_RECENT_FOLDERS
            except json.JSONDecodeError as e:
                print(f"Error reading GUI config (corrupted JSON?): {e}")
                self.recent_folders_list = []
            except Exception as e:
                print(f"Error loading GUI config: {e}")
                self.recent_folders_list = []

    def _exit_app(self):
        self._save_gui_config()
        self._save_progress() # Always save progress on exit
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ConvergenceJSONTool(root)
    root.protocol("WM_DELETE_WINDOW", app._exit_app) # Handle window close button
    root.mainloop()