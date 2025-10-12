import os
import gc
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable, Dict, Any
import torch
import numpy as np
from PIL import Image, ImageTk
from decord import VideoReader, cpu

# Import release_cuda_memory from the util module
from .stereocrafter_util import Tooltip, logger, release_cuda_memory

class VideoPreviewer(ttk.Frame):
    """
    A self-contained Tkinter widget for previewing video processing results.

    This module handles:
    - Displaying a preview image on a scrollable canvas.
    - Navigating through a list of videos.
    - Scrubbing through the timeline of the current video.
    - Loading single frames from multiple source videos.
    - Calling a user-provided processing function to generate the preview.
    """
    def __init__(self, parent, processing_callback: Callable, find_sources_callback: Optional[Callable] = None, help_data: Dict[str, str] = None, **kwargs):
        """
        Initializes the VideoPreviewer frame.

        Args:
            parent: The parent tkinter widget.
            processing_callback (Callable): A function that takes two arguments:
                - A dictionary of source frames, e.g., {'inpainted': tensor, 'original': tensor}.
                - A dictionary of parameters from the main GUI.
                It should return a PIL Image to be displayed.
            find_sources_callback (Callable, optional): A function that returns a list of
                dictionaries, where each dict maps a source name to a file path.
            help_data (Dict[str, str], optional): A dictionary of help texts for tooltips.
        """
        super().__init__(parent, **kwargs)
        self.parent = parent
        self.processing_callback = processing_callback
        self.help_data = help_data if help_data else {}
        self.find_sources_callback = find_sources_callback

        # --- State ---
        self.source_readers: Dict[str, VideoReader] = {}
        self.video_list: list[Dict[str, str]] = []
        self.current_video_index: int = -1
        self.current_params: Dict[str, Any] = {}
        self.pil_image_for_preview: Optional[Image.Image] = None
        self.preview_image_tk: Optional[ImageTk.PhotoImage] = None
        self.wiggle_after_id: Optional[str] = None

        # --- GUI Variables ---
        self.frame_scrubber_var = tk.DoubleVar(value=0)
        self.video_jump_to_var = tk.StringVar(value="1")
        self.video_status_label_var = tk.StringVar(value="Video: 0 / 0")
        self.frame_label_var = tk.StringVar(value="Frame: 0 / 0")
        self.preview_size_var = tk.StringVar(value="1000") # Default, can be overridden

        self._create_widgets()

    def _create_hover_tooltip(self, widget, help_key):
        """Creates a mouse-over tooltip for the given widget."""
        if help_key in self.help_data:
            Tooltip(widget, self.help_data[help_key])

    def _create_widgets(self):
        """Creates and lays out all the widgets for the previewer."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Canvas with scrollbars for the image
        self.preview_canvas = tk.Canvas(self)
        v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.preview_canvas.yview)
        h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self.preview_canvas.xview)
        self.preview_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.preview_canvas.bind("<Configure>", lambda e: self._update_preview_layout())

        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.v_scrollbar = v_scrollbar
        self.h_scrollbar = h_scrollbar

        self.preview_inner_frame = ttk.Frame(self.preview_canvas)
        self.preview_canvas_window_id = self.preview_canvas.create_window((0, 0), window=self.preview_inner_frame, anchor="nw")
        self.preview_label = ttk.Label(self.preview_inner_frame, text="Load a video list to see preview", anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        # Scrubber Frame
        scrubber_frame = ttk.Frame(self)
        scrubber_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        scrubber_frame.grid_columnconfigure(1, weight=1)

        self.frame_label = ttk.Label(scrubber_frame, textvariable=self.frame_label_var, width=15)
        self.frame_label.grid(row=0, column=0, padx=5)
        self.frame_scrubber = ttk.Scale(scrubber_frame, from_=0, to=0, variable=self.frame_scrubber_var, orient="horizontal")
        self.frame_scrubber.grid(row=0, column=1, sticky="ew")
        self.frame_scrubber.bind("<ButtonRelease-1>", self.on_slider_release)
        self.frame_scrubber.configure(command=self.on_scrubber_move)

        # Video Navigation Frame
        preview_button_frame = ttk.Frame(self)
        preview_button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=5)

        # Add Preview Source dropdown
        ttk.Label(preview_button_frame, text="Preview Source:").pack(side="left", padx=(0, 5))
        self.preview_source_combo = ttk.Combobox(preview_button_frame, state="readonly", width=18)
        self.preview_source_combo.pack(side="left", padx=5)
        self.preview_source_combo.bind("<<ComboboxSelected>>", self.on_slider_release)
        self._create_hover_tooltip(self.preview_source_combo, "preview_source")

        self.load_preview_button = ttk.Button(preview_button_frame, text="Load/Refresh List", command=self._handle_load_refresh, width=20)
        self.load_preview_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.load_preview_button, "load_refresh_list")

        self.prev_video_button = ttk.Button(preview_button_frame, text="< Prev", command=lambda: self._nav_preview_video(-1))
        self.prev_video_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.prev_video_button, "prev_video")

        self.next_video_button = ttk.Button(preview_button_frame, text="Next >", command=lambda: self._nav_preview_video(1))
        self.next_video_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.next_video_button, "next_video")

        ttk.Label(preview_button_frame, text="Jump to:").pack(side="left", padx=(15, 2))
        self.video_jump_entry = ttk.Entry(preview_button_frame, textvariable=self.video_jump_to_var, width=5)
        self.video_jump_entry.pack(side="left")
        self.video_jump_entry.bind("<Return>", self._jump_to_video)
        self._create_hover_tooltip(self.video_jump_entry, "jump_to_video")
        ttk.Label(preview_button_frame, textvariable=self.video_status_label_var).pack(side="left", padx=5)

    def set_parameters(self, params: Dict[str, Any]):
        """
        Receives a dictionary of parameters from the main GUI.
        Triggers a preview update if the parameters have changed.
        """
        if params != self.current_params:
            self.current_params = params
            self.update_preview()

    def set_preview_source_options(self, options: list):
        """Sets the available options for the preview source dropdown."""
        self.preview_source_combo['values'] = options
        # The variable is now controlled by the parent GUI, so we just check its current value
        if self.preview_source_combo.cget('textvariable') and self.preview_source_combo.get() not in options:
            self.preview_source_combo.set(options[0] if options else "Blended Image")


    def _handle_load_refresh(self):
        """Internal handler for the 'Load/Refresh List' button."""
        if self.find_sources_callback:
            self.load_video_list(find_sources_callback=self.find_sources_callback)
        else:
            logger.error("VideoPreviewer: 'find_sources_callback' was not provided during initialization. Cannot load video list.")
            messagebox.showerror("Initialization Error", "The 'find_sources_callback' was not provided to the previewer.")

    def load_video_list(self, find_sources_callback: Callable):
        """
        Loads a list of video sources to be previewed.

        Args:
            find_sources_callback (Callable): A function that returns a list of dictionaries.
        """
        self._clear_preview_resources()

        self.video_list = find_sources_callback()

        if not self.video_list:
            messagebox.showwarning("Not Found", "No valid source videos found.")
            self.current_video_index = -1
            self._update_nav_controls()
            return

        self.current_video_index = 0
        self._load_preview_by_index(self.current_video_index)

    def update_preview(self):
        """The main preview generation function."""
        if not self.source_readers:
            return

        self._stop_wigglegram_animation()
        self.load_preview_button.config(text="LOADING...", style="Loading.TButton")
        self.parent.update_idletasks()

        try:
            frame_idx = int(self.frame_scrubber_var.get())

            # Load the single frame from each source reader
            source_frames = {}
            for key, reader in self.source_readers.items():
                if reader:
                    frame_np = reader.get_batch([frame_idx]).asnumpy()
                    frame_tensor = torch.from_numpy(frame_np).permute(0, 3, 1, 2).float() / 255.0
                    source_frames[key] = frame_tensor # Keep batch dim: [1, C, H, W]

            # Call the user-provided processing function
            self.pil_image_for_preview = self.processing_callback(source_frames, self.current_params)

            # If the callback returned None, check if it was because a wigglegram was started.
            # If not, then it's a genuine error.
            if self.pil_image_for_preview is None and self.wiggle_after_id is None:
                raise ValueError("Processing callback returned None.")
            
            # --- FIX: If wigglegram started, the callback returns None. Exit here. ---
            if self.wiggle_after_id is not None:
                return # The wigglegram animation loop will handle the display.
            # --- END FIX ---

            # Resize for display if too large, maintaining aspect ratio
            max_size = int(self.preview_size_var.get())
            display_image = self.pil_image_for_preview.copy()
            if display_image.height > max_size or display_image.width > max_size:
                display_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            self.preview_image_tk = ImageTk.PhotoImage(display_image)
            # --- FIX: Attach the image reference to the widget to prevent garbage collection ---
            self.preview_label.config(image=self.preview_image_tk, text="")
            self.preview_label.image = self.preview_image_tk
            # --- END FIX ---
            self._update_preview_layout()

        except Exception as e:
            logger.error(f"Error updating preview: {e}", exc_info=True)
            self.preview_label.config(image=None, text=f"Error:\n{e}")
        finally:
            release_cuda_memory()
            self.load_preview_button.config(text="Load/Refresh List", style="TButton")

    def on_slider_release(self, event):
        """Called when a slider is released. Updates the preview."""
        self._stop_wigglegram_animation()
        if self.source_readers:
            self.update_preview()

    def on_scrubber_move(self, value):
        """Called continuously as the frame scrubber moves to update the label."""
        frame_idx = int(float(value))
        total_frames = int(self.frame_scrubber.cget("to")) + 1
        self.frame_label_var.set(f"Frame: {frame_idx + 1} / {total_frames}")

    def _clear_preview_resources(self):
        """Closes all preview-related video readers and clears the preview display."""
        self._stop_wigglegram_animation()

        for key in list(self.source_readers.keys()):
            if self.source_readers[key]:
                del self.source_readers[key]
        self.source_readers.clear()

        # --- FIX: Clear the widget's image reference before setting the image to None ---
        self.preview_label.config(image=None, text="Load a video list to see preview")
        self.preview_label.image = None
        # --- END FIX ---
        self.preview_image_tk = None
        self.pil_image_for_preview = None
        gc.collect()
        logger.info("Preview resources and file handles have been released.")

    def _load_preview_by_index(self, index: int):
        """Loads a specific video from the preview list by its index."""
        self._clear_preview_resources()

        if not (0 <= index < len(self.video_list)):
            return

        self.current_video_index = index
        self._update_nav_controls()

        source_paths = self.video_list[index]
        base_name = os.path.basename(next(iter(source_paths.values()))) # Get name from first path

        self.load_preview_button.config(text="LOADING...", style="Loading.TButton")
        self.parent.update_idletasks()

        try:
            # Initialize VideoReader for each source path
            num_frames = -1
            for key, path in source_paths.items():
                if path and os.path.exists(path):
                    reader = VideoReader(path, ctx=cpu(0))
                    if num_frames == -1:
                        num_frames = len(reader)
                    elif num_frames != len(reader):
                        raise ValueError(f"Frame count mismatch between sources for {base_name}")
                    self.source_readers[key] = reader
                else:
                    self.source_readers[key] = None
                    logger.warning(f"Source '{key}' not found for {base_name} at path: {path}")

            if num_frames <= 0:
                raise ValueError("Video has no frames or could not be loaded.")

            # Configure the scrubber
            self.frame_scrubber.config(to=num_frames - 1)
            self.frame_scrubber_var.set(0)
            self.on_scrubber_move(0) # Update label

            self.update_preview()

        except Exception as e:
            messagebox.showerror("Preview Load Error", f"Failed to load files for preview:\n\n{e}")
            logger.error("Preview load failed", exc_info=True)
        finally:
            self.load_preview_button.config(text="Load/Refresh List", style="TButton")

    def _nav_preview_video(self, direction: int):
        """Navigate to the previous or next video in the preview list."""
        if not self.video_list:
            return
        new_index = self.current_video_index + direction
        if 0 <= new_index < len(self.video_list):
            self._load_preview_by_index(new_index)

    def _jump_to_video(self, event=None):
        """Jump to a specific video number in the preview list."""
        if not self.video_list:
            return
        try:
            target_index = int(self.video_jump_to_var.get()) - 1
            if 0 <= target_index < len(self.video_list):
                self._load_preview_by_index(target_index)
            else:
                messagebox.showwarning("Out of Range", f"Please enter a number between 1 and {len(self.video_list)}.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    def _update_nav_controls(self):
        """Updates the state and labels of the video navigation controls."""
        total_videos = len(self.video_list)
        current_index = self.current_video_index

        self.video_status_label_var.set(f"Video: {current_index + 1} / {total_videos}" if total_videos > 0 else "Video: 0 / 0")
        self.video_jump_to_var.set(str(current_index + 1) if total_videos > 0 else "1")

        self.prev_video_button.config(state="normal" if current_index > 0 else "disabled")
        self.next_video_button.config(state="normal" if 0 <= current_index < total_videos - 1 else "disabled")
        self.video_jump_entry.config(state="normal" if total_videos > 0 else "disabled")

    def _update_preview_layout(self):
        """Centers the image if it's smaller than the canvas, and hides/shows scrollbars."""
        if not hasattr(self, 'preview_canvas') or self.pil_image_for_preview is None:
            if hasattr(self, 'v_scrollbar'): self.v_scrollbar.grid_remove()
            if hasattr(self, 'h_scrollbar'): self.h_scrollbar.grid_remove()
            return

        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        
        # Use the PhotoImage size for layout, not the original PIL image
        img_w = self.preview_image_tk.width()
        img_h = self.preview_image_tk.height()

        v_scroll_needed = img_h > canvas_h
        h_scroll_needed = img_w > canvas_w

        if v_scroll_needed: self.v_scrollbar.grid()
        else: self.v_scrollbar.grid_remove()
        if h_scroll_needed: self.h_scrollbar.grid()
        else: self.h_scrollbar.grid_remove()

        x = max(0, (canvas_w - img_w) // 2)
        y = max(0, (canvas_h - img_h) // 2)
        
        self.preview_canvas.coords(self.preview_canvas_window_id, x, y)
        self.preview_inner_frame.update_idletasks()
        self.preview_canvas.config(scrollregion=self.preview_canvas.bbox("all"))

    def _stop_wigglegram_animation(self):
        if self.wiggle_after_id:
            self.parent.after_cancel(self.wiggle_after_id)
            self.wiggle_after_id = None
        if hasattr(self, 'wiggle_left_tk'): del self.wiggle_left_tk
        if hasattr(self, 'wiggle_right_tk'): del self.wiggle_right_tk

    def _start_wigglegram_animation(self, left_frame: torch.Tensor, right_frame: torch.Tensor):
        """Starts the wigglegram animation loop."""
        self._stop_wigglegram_animation()

        max_size = int(self.preview_size_var.get())
        
        left_np = (left_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        left_pil = Image.fromarray(left_np)
        if left_pil.height > max_size or left_pil.width > max_size:
            left_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        self.wiggle_left_tk = ImageTk.PhotoImage(left_pil)

        right_np = (right_frame.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        right_pil = Image.fromarray(right_np)
        if right_pil.height > max_size or right_pil.width > max_size:
            right_pil.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        self.wiggle_right_tk = ImageTk.PhotoImage(right_pil)

        self._wiggle_step(True)

    def _wiggle_step(self, show_left: bool):
        """A single step in the wigglegram animation."""
        if not hasattr(self, 'wiggle_left_tk'): return # Stop if resources were cleared
        current_image = self.wiggle_left_tk if show_left else self.wiggle_right_tk
        self.preview_label.config(image=current_image)
        self.preview_label.image = current_image # Prevent garbage collection
        self.wiggle_after_id = self.parent.after(60, self._wiggle_step, not show_left)

    def save_preview_frame(self):
        """Saves the current preview image to a file."""
        if self.pil_image_for_preview is None:
            messagebox.showwarning("No Preview", "There is no preview image to save.")
            return

        default_filename = "preview_frame.png"
        if self.current_video_index != -1:
            source_paths = self.video_list[self.current_video_index]
            base_name = os.path.splitext(os.path.basename(next(iter(source_paths.values()))))[0]
            frame_num = int(self.frame_scrubber_var.get())
            default_filename = f"{base_name}_frame_{frame_num:05d}.png"

        filepath = filedialog.asksaveasfilename(
            title="Save Preview Frame As...",
            initialfile=default_filename,
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )

        if filepath:
            try:
                self.pil_image_for_preview.save(filepath)
                logger.info(f"Preview frame saved to: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save preview frame: {e}", exc_info=True)
                messagebox.showerror("Save Error", f"An error occurred while saving the image:\n{e}")

    def cleanup(self):
        """Public method to be called when the parent GUI is closing."""
        self._clear_preview_resources()