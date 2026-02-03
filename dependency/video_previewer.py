import os
import json
import gc
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional, Callable, Dict, Any, Union
import torch
import numpy as np
import subprocess
from PIL import Image, ImageTk, ImageDraw, ImageFont
from decord import VideoReader, cpu

# Import release_cuda_memory from the util module
from .stereocrafter_util import Tooltip, logger, release_cuda_memory, get_video_stream_info

VERSION = "26-01-30.0"

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
    def __init__(
            self,
            parent,
            processing_callback: Callable,
            find_sources_callback: Optional[Callable] = None,
            get_params_callback: Optional[Callable] = None,
            help_data: Dict[str, str] = None,
            preview_size_var: Optional[tk.StringVar] = None,
            resize_callback: Optional[Callable] = None,
            update_clip_callback: Optional[Callable] = None,
            on_clip_navigate_callback: Optional[Callable] = None,
            **kwargs,
        ):
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
            get_params_callback (Callable, optional): A function that returns the current
                dictionary of parameters from the main GUI.
            help_data (Dict[str, str], optional): A dictionary of help texts for tooltips.
            preview_size_var (tk.StringVar, optional): The variable from the parent GUI to control preview size.
            resize_callback (Callable, optional): A function to call to ask the parent window to resize itself.
        """
        super().__init__(parent, **kwargs)
        self.parent = parent
        
        # Depth-map decode state (10-bit+ aware)
        self._depth_path: Optional[str] = None
        self._depth_bit_depth: int = 8
        self._depth_is_high_bit: bool = False
        self._depth_native_w: Optional[int] = None
        self._depth_native_h: Optional[int] = None
        self._depth_msb_shift: Optional[int] = None
        self.processing_callback = processing_callback
        self.help_data = help_data if help_data else {}
        self.find_sources_callback = find_sources_callback
        self.get_params_callback = get_params_callback
        self.preview_size_var = preview_size_var # Store the passed-in variable
        self.resize_callback = resize_callback # Store the resize callback
        self.update_clip_callback = update_clip_callback
        self.on_clip_navigate_callback = on_clip_navigate_callback

        # --- State ---
        self.source_readers: Dict[str, Optional[VideoReader]] = {}
        self.video_list: list[Dict[str, str]] = []
        self.current_video_index: int = -1
        self.current_params: Dict[str, Any] = {}
        self.pil_image_for_preview: Optional[Image.Image] = None
        self.preview_image_tk: Optional[ImageTk.PhotoImage] = None
        self.wiggle_after_id: Optional[str] = None
        self.root_window = self.parent.winfo_toplevel() 
        self.last_loaded_video_path: Optional[str] = None
        self.last_loaded_frame_index: int = 0 
        # --- Playback (preview-only) ---
        self._play_after_id: Optional[str] = None
        self._is_playing: bool = False
        self._play_step: int = 1  # 1=Play, 5=Fast Forward


        self.loop_playback_var = tk.BooleanVar(value=False)
        # Restore Loop state from the main GUI config file (lightweight; safe if missing)
        try:
            # Prefer the canonical .splatcfg default config file; fall back to legacy .json
            cfg_path = "config_splat.splatcfg" if os.path.exists("config_splat.splatcfg") else ("config_splat.json" if os.path.exists("config_splat.json") else "config_splat.splatcfg")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    _cfg = json.load(f) or {}
                if "loop_playback" in _cfg:
                    self.loop_playback_var.set(bool(_cfg.get("loop_playback", False)))
        except Exception:
            pass
        # --- GUI Variables ---
        self.frame_scrubber_var = tk.DoubleVar(value=0)
        self.video_jump_to_var = tk.StringVar(value="1")
        self.video_status_label_var = tk.StringVar(value="Video: 0 / 0")
        self.frame_label_var = tk.StringVar(value="Frame: 0 / 0")
        self._is_dragging = False

        # Crosshair overlay state (preview only). Controlled by parent GUI.
        self.crosshair_enabled = False
        self.crosshair_white = False
        self.crosshair_multi = False  # show additional bullseyes/dots (preview only)
        self.depth_pop_depth_pct = None  # background separation (% of width)
        self.depth_pop_pop_pct = None    # foreground separation (% of width)
        self.depth_pop_enabled = False    # show Depth/Pop readout (preview only)
        self._dp_total_max_seen = None
        self._dp_total_max_video_index = None
        self._dp_signature = None

        self._create_widgets()

    def cleanup(self):
        """Public method to be called when the parent GUI is closing."""
        # Persist Loop state back into the main GUI config file (merge; do not clobber)
        try:
            # Prefer the canonical .splatcfg default config file; fall back to legacy .json
            cfg_path = "config_splat.splatcfg" if os.path.exists("config_splat.splatcfg") else ("config_splat.json" if os.path.exists("config_splat.json") else "config_splat.splatcfg")
            _cfg = {}
            if os.path.exists(cfg_path):
                with open(cfg_path, "r") as f:
                    _cfg = json.load(f) or {}
            _cfg["loop_playback"] = bool(self.loop_playback_var.get())
            with open(cfg_path, "w") as f:
                json.dump(_cfg, f, indent=4)
        except Exception:
            pass
        self._clear_preview_resources()

    def _clear_preview_resources(self):
        """Closes all preview-related video readers and clears the preview display."""
        self._stop_playback()
        self._stop_wigglegram_animation()

        for key in list(self.source_readers.keys()):
            if self.source_readers[key]:
                del self.source_readers[key]
        self.source_readers.clear()

        # --- FIX: Create a dummy image to hold the place, preventing TclError ---
        # This is the most robust way to clear the image in Tkinter without race conditions.
        self._dummy_image = ImageTk.PhotoImage(Image.new('RGBA', (1, 1), (0,0,0,0)))
        self.preview_label.config(image=self._dummy_image, text="Load a video list to see preview")
        self.preview_label.image = self._dummy_image
        self.preview_image_tk = None
        # --- END FIX ---
        self.pil_image_for_preview = None


        # Reset depth-map decode state

        self._depth_path = None

        self._depth_bit_depth = 8

        self._depth_is_high_bit = False

        self._depth_native_w = None

        self._depth_native_h = None

        self._depth_msb_shift = None

        gc.collect()
        logger.debug("Preview resources and file handles have been released.")

    def _create_hover_tooltip(self, widget, help_key, tooltip_info: Optional[str] = None):
        """Creates a mouse-over tooltip for the given widget."""
        if help_key in self.help_data:
            Tooltip(widget, self.help_data[help_key])
        elif tooltip_info:
            Tooltip(widget, tooltip_info)

    def _create_widgets(self):
        """Creates and lays out all the widgets for the previewer."""
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Canvas with scrollbars for the image
        self.preview_canvas = tk.Canvas(self)
        v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._on_preview_vscroll)
        h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self._on_preview_hscroll)
        self.preview_canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
        self.preview_canvas.bind("<Configure>", lambda e: self._update_preview_layout())
        # Keep crosshair centered when scrolling (does not override scroll behavior)


        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar.grid(row=1, column=0, sticky="ew")

        self.v_scrollbar = v_scrollbar
        self.h_scrollbar = h_scrollbar

        self.preview_inner_frame = ttk.Frame(self.preview_canvas)
        self.preview_canvas_window_id = self.preview_canvas.create_window((0, 0), window=self.preview_inner_frame, anchor="nw")
        self.preview_label = ttk.Label(self.preview_inner_frame, text="Load a video list to see preview", anchor="center")
        self.preview_label.pack(fill="both", expand=True)
        
        # self.preview_canvas.itemconfig(self.preview_canvas_window_id, tags=("content_drag_tag",))
        # # Start: Call scan_mark and return break
        # self.preview_label.bind("<ButtonPress-1>", 
        #                         lambda e: (self.preview_canvas.scan_mark(e.x, e.y), "break")[1])
        
        # # Drag: Call scan_dragto and return break
        # self.preview_label.bind("<B1-Motion>", 
        #                         lambda e: (self.preview_canvas.scan_dragto(e.x, e.y, gain=1), "break")[1])
        
        # # End: Call the method to clear the cursor
        # self.preview_label.bind("<ButtonRelease-1>", self._end_drag_scroll)

        # Scrubber Frame
        scrubber_frame = ttk.Frame(self)
        scrubber_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=2)
        scrubber_frame.grid_columnconfigure(1, weight=1)

        self.frame_label = ttk.Label(scrubber_frame, textvariable=self.frame_label_var, width=15)
        self.frame_label.grid(row=0, column=0, padx=5)
        self.frame_scrubber = ttk.Scale(scrubber_frame, from_=0, to=0, variable=self.frame_scrubber_var, orient="horizontal")
        self.frame_scrubber.grid(row=0, column=1, sticky="ew")
        self.frame_scrubber.bind("<ButtonRelease-1>", self.on_slider_release)
        self.frame_scrubber.bind("<Button-1>", self._on_scrubber_trough_click)
        self.frame_scrubber.configure(command=self.on_scrubber_move)
        
        # Video Navigation Frame
        preview_button_frame = ttk.Frame(self)
        preview_button_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=2)
        
        # Bindings are placed on the top-level window for global detection
        self.root_window.bind('<Left>', self._key_jump_frames, add='+')
        self.root_window.bind('<Right>', self._key_jump_frames, add='+')
        self.root_window.bind('<Shift-Left>', self._key_jump_frames, add='+')
        self.root_window.bind('<Shift-Right>', self._key_jump_frames, add='+')
        self.root_window.bind('<Control-Left>', self._key_jump_clips, add='+')
        self.root_window.bind('<Control-Right>', self._key_jump_clips, add='+')
        # Optional: Up/Down to navigate between clips (only if not already bound elsewhere)
        if not self.root_window.bind('<Up>'):
            self.root_window.bind('<Up>', self._key_nav_clips_updown, add='+')
        if not self.root_window.bind('<Down>'):
            self.root_window.bind('<Down>', self._key_nav_clips_updown, add='+')
        # Playback shortcuts (preview-only)
        self.root_window.bind('<space>', self._key_toggle_play_pause, add='+')
        self.root_window.bind('<Shift-space>', self._key_shift_space_fast_forward, add='+')

        logger.debug("Global key bindings for frame jumping installed on root window.")

        # Add Preview Source dropdown
        lbl_preview_source = ttk.Label(preview_button_frame, text="Preview Source:")
        lbl_preview_source.pack(side="left", padx=(0, 5))
        self.preview_source_combo = ttk.Combobox(preview_button_frame, state="readonly", width=18)
        self.preview_source_combo.pack(side="left", padx=5)
        # Prevent this combobox from stealing focus/keyboard shortcuts (space/enter)
        self.preview_source_combo.configure(takefocus=False)
        self.preview_source_combo.bind("<<ComboboxSelected>>", lambda e: (self.on_slider_release(e), self.preview_canvas.focus_set()))
        self.preview_source_combo.bind("<space>", lambda e: (self._key_toggle_play_pause(e) or "break"))
        self.preview_source_combo.bind("<Shift-space>", lambda e: (self._key_shift_space_fast_forward(e) or "break"))
        self.preview_source_combo.bind("<Return>", lambda e: (self.root_window.focus_set() or self.root_window.event_generate("<Return>") or "break"))
        self.preview_source_combo.bind("<KP_Enter>", lambda e: (self.root_window.focus_set() or self.root_window.event_generate("<Return>") or "break"))
        tip_preview_source = "Select which image layer to display in the preview window for diagnostic purposes."
        self._create_hover_tooltip(lbl_preview_source, "preview_source", tip_preview_source)
        self._create_hover_tooltip(self.preview_source_combo, "preview_source", tip_preview_source)

        self.load_preview_button = ttk.Button(preview_button_frame, text="Load/Refresh List", command=self._handle_load_refresh, width=20, takefocus=False)
        self.load_preview_button.pack(side="left", padx=5)
        tip_load_refresh_list = "Scans the 'Inpainted Video Folder' for valid files and loads the first one for preview."
        self._create_hover_tooltip(self.load_preview_button, "load_refresh_list", tip_load_refresh_list)

        self.prev_video_button = ttk.Button(preview_button_frame, text="< Prev", command=lambda: self._nav_preview_video(-1), takefocus=False)
        self.prev_video_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.prev_video_button, "prev_video", "Load the previous video in the list for preview.")

        self.next_video_button = ttk.Button(preview_button_frame, text="Next >", command=lambda: self._nav_preview_video(1), takefocus=False)
        self.next_video_button.pack(side="left", padx=5)
        self._create_hover_tooltip(self.next_video_button, "next_video", "Load the next video in the list for preview.")

        lbl_video_jump_entry = ttk.Label(preview_button_frame, text="Jump to:")
        lbl_video_jump_entry.pack(side="left", padx=(15, 2))
        self.video_jump_entry = ttk.Entry(preview_button_frame, textvariable=self.video_jump_to_var, width=5)
        self.video_jump_entry.pack(side="left")
        self.video_jump_entry.bind("<Return>", self._jump_to_video)
        lbl_video_jump_info = ttk.Label(preview_button_frame, textvariable=self.video_status_label_var)
        lbl_video_jump_info.pack(side="left", padx=5)
        tip_jump_to_video = "Enter a video number and press Enter to jump directly to it in the list."
        tip_jump_info ="Displys which frame number from total number of frames. (Current_frame/Total_frames)"
        self._create_hover_tooltip(lbl_video_jump_entry, "jump_to_video", tip_jump_to_video)
        self._create_hover_tooltip(self.video_jump_entry, "jump_to_video", tip_jump_to_video)
        self._create_hover_tooltip(lbl_video_jump_info, "jump_to_info", tip_jump_info)

        # --- NEW: Playback controls (preview-only) ---
        self.play_pause_button = ttk.Button(preview_button_frame, text="▶", width=3, command=self._toggle_play_pause, takefocus=False)
        self.play_pause_button.pack(side="left", padx=(5, 2))
        tip_play_pause = "Play/Pause (frame-by-frame). Shortcut: Spacebar"
        self._create_hover_tooltip(self.play_pause_button, "preview_play_pause", tip_play_pause)

        self.fast_forward_button = ttk.Button(preview_button_frame, text=">>", width=3, command=self._toggle_fast_forward, takefocus=False)
        self.fast_forward_button.pack(side="left", padx=(2, 5))
        tip_fast_forward = "Fast Forward (step 5 frames). Shortcut: Shift+Spacebar (Spacebar pauses)"
        self._create_hover_tooltip(self.fast_forward_button, "preview_fast_forward", tip_fast_forward)
        # Prevent focused buttons from also consuming Space/Return via ttk default bindings.
        # This avoids double-toggling when the user clicks Fast Forward (button gains focus) then presses Space.
        self.play_pause_button.bind("<space>", lambda e: (self._toggle_play_pause(), "break")[1])
        self.play_pause_button.bind("<Return>", lambda e: (self._toggle_play_pause(), "break")[1])
        self.fast_forward_button.bind("<space>", lambda e: (self._toggle_play_pause(), "break")[1])
        self.fast_forward_button.bind("<Return>", lambda e: (self._toggle_play_pause(), "break")[1])
        # Loop indicator (clickable). Avoid ttk color limitations by using a tk.Label.
        self.loop_label = tk.Label(preview_button_frame, text="🔁", cursor="hand2")
        self.loop_label.pack(side="left", padx=(4, 4))
        self.loop_label.bind("<Button-1>", self._toggle_loop)
        tip_loop = "Loop playback. When reaching the end, wrap to the first frame and continue."
        self._create_hover_tooltip(self.loop_label, "preview_loop", tip_loop)
        self._update_loop_indicator()


        
        # --- MODIFIED: Add Preview Size Combobox (Percentage Scale) ---
        PERCENTAGE_VALUES = [
                "250%", "240%", "230%", "220%", "210%", "200%", "190%", "180%", "170%", "160%", 
                "150%", "145%", "140%", "135%", "130%", "125%", "120%", "115%", "110%", "105%", 
                "100%", "95%", "90%", "85%", "80%", "75%", "70%", "65%", "60%", "55%", "50%", "25%"
            ]
        
        lbl_preview_scale = ttk.Label(preview_button_frame, text="Preview Scale:")
        lbl_preview_scale.pack(side="left", padx=(10, 5))
        tip_preview_scale = "Select the size of the video preview. Larger images may impact performance."
        self._create_hover_tooltip(lbl_preview_scale, "preview_scale", tip_preview_scale)
        
        self.preview_size_combo = ttk.Combobox(
            preview_button_frame, 
            textvariable=self.preview_size_var, 
            values=PERCENTAGE_VALUES, 
            state="readonly", # Make it selection-only
            width=5,
            takefocus=False
        )
        self.preview_size_combo.pack(side="left")
        self._create_hover_tooltip(self.preview_size_combo, "preview_scale", tip_preview_scale)
        
        # We need to explicitly bind the ComboboxSelected event to update the preview
        self.preview_size_combo.bind("<<ComboboxSelected>>", self.on_slider_release)
        
        self._create_hover_tooltip(self.preview_size_combo, "preview_size", tip_preview_scale)
        
        # Re-assign to a variable name used later for disabling/enabling
        self.preview_size_entry = self.preview_size_combo 
        # --- END MODIFIED ---

        # --- NEW: Store widgets to be disabled ---
        self.widgets_to_disable = [self.load_preview_button, self.prev_video_button, self.next_video_button, self.play_pause_button, self.fast_forward_button,
                                   self.video_jump_entry, self.frame_scrubber, self.preview_source_combo, self.preview_size_combo]

        # --- [START OF ADDITION] ZOOM & DRAG INTERACTION BINDINGS ---
        self.preview_canvas.bind("<Enter>", lambda e: self.preview_canvas.focus_set())
        
        # Universal Zoom (Mousewheel)
        for w in [self.preview_canvas, self.preview_label]:
            w.bind("<MouseWheel>", self._handle_zoom)
            w.bind("<Button-4>", self._handle_zoom)
            w.bind("<Button-5>", self._handle_zoom)
            # Right Click to reset zoom to 100%
            w.bind("<Button-3>", lambda e: (self.preview_size_var.set("100%"), self.on_slider_release(None)))

        # Universal Drag (Left Click and Middle Click)
        for b in ["<ButtonPress-1>", "<ButtonPress-2>"]:
            self.preview_label.bind(b, self._start_drag_scroll)
        for b in ["<B1-Motion>", "<B2-Motion>"]:
            self.preview_label.bind(b, self._drag_scroll)
        for b in ["<ButtonRelease-1>", "<ButtonRelease-2>"]:
            self.preview_label.bind(b, self._end_drag_scroll)

    def _start_drag_scroll(self, event):
        """Initiates panning logic if image is larger than the window."""
        if self.v_scrollbar.winfo_ismapped() or self.h_scrollbar.winfo_ismapped():
            self._is_dragging = True
            self.preview_canvas.config(cursor="fleur")
            self.preview_canvas.scan_mark(int(event.x), int(event.y))

    def _end_drag_scroll(self, event):
        """Resets the cursor and dragging state when the mouse button is released."""
        self._is_dragging = False
        # Remove the 'fleur' panning cursor
        self.preview_canvas.config(cursor="")
        logger.debug("_end_drag_scroll: Panning operation concluded.")

    def _drag_scroll(self, event):
        """Standard Canvas panning using gain=1 for 1:1 mouse movement."""
        if self._is_dragging:
            self.preview_canvas.scan_dragto(int(event.x), int(event.y), gain=1)

    def _handle_zoom(self, event):
        """Standard Mousewheel Zoom stepping through the percentage list."""
        if not self.preview_size_var: return
        vals = ["250%", "240%", "230%", "220%", "210%", "200%", "190%", "180%", "170%", "160%", 
        "150%", "145%", "140%", "135%", "130%", "125%", "120%", "115%", "110%", "105%", 
        "100%", "95%", "90%", "85%", "80%", "75%", "70%", "65%", "60%", "55%", "50%", "25%"
    ]
        curr = self.preview_size_var.get()
        try: idx = vals.index(curr)
        except ValueError: idx = vals.index("100%")
        
        delta = 1 if (event.num == 4 or (hasattr(event, 'delta') and event.delta > 0)) else -1
        new_idx = max(0, min(len(vals) - 1, idx - delta))
        
        if new_idx != idx:
            self.preview_size_var.set(vals[new_idx])
            if hasattr(self.parent, 'on_slider_release'): self.parent.on_slider_release(None)
            else: self.on_slider_release(None)
                
    def _handle_load_refresh(self):
        """Internal handler for the 'Load/Refresh List' button."""
        self._stop_playback()
        if self.find_sources_callback:
            self.load_video_list(find_sources_callback=self.find_sources_callback)
        else:
            logger.error("VideoPreviewer: 'find_sources_callback' was not provided during initialization. Cannot load video list.")
            messagebox.showerror("Initialization Error", "The 'find_sources_callback' was not provided to the previewer.")

    def _jump_to_video(self, event=None):
        """Jump to a specific video number in the preview list."""
        self._stop_playback()
        if not self.video_list:
            return
        
        if self.on_clip_navigate_callback:
            self.on_clip_navigate_callback()

        try:
            target_index = int(self.video_jump_to_var.get()) - 1
            if 0 <= target_index < len(self.video_list):
                self._load_preview_by_index(target_index)
            else:
                messagebox.showwarning("Out of Range", f"Please enter a number between 1 and {len(self.video_list)}.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number.")

    def _key_jump_clips(self, event):
        """Handler for Ctrl+Left/Right arrow keys to jump between clips."""
        if not self.video_list:
            return

        direction = 0
        if event.keysym == "Left":
            direction = -1
        elif event.keysym == "Right":
            direction = 1
        else:
            return # Should not happen

        # Call the existing navigation function
        self._nav_preview_video(direction)


    def _key_nav_clips_updown(self, event):
        """Handler for Up/Down arrow keys to navigate between clips."""
        if not self.video_list:
            return

        # Don't hijack Up/Down inside text/entry/combobox widgets (let them behave normally)
        try:
            wclass = event.widget.winfo_class()
        except Exception:
            wclass = ""
        if wclass in ("TEntry", "Entry", "TCombobox", "Combobox", "TSpinbox", "Spinbox", "Text"):
            return

        if event.keysym == "Up":
            self._nav_preview_video(1)
            return "break"
        elif event.keysym == "Down":
            self._nav_preview_video(-1)
            return "break"
        return
    def _jump_frames_by(self, delta: int):
        """Core frame-stepping logic shared by arrow keys and playback controls."""
        if not self.source_readers:
            return

        current_frame = int(self.frame_scrubber_var.get())
        total_frames = int(self.frame_scrubber.cget("to")) + 1

        new_frame = current_frame + int(delta)

        # Clamp the new frame index
        new_frame = max(0, min(new_frame, total_frames - 1))

        if new_frame != current_frame:
            self.frame_scrubber_var.set(new_frame)
            self.on_scrubber_move(new_frame)  # Update label
            self.update_preview()  # Update display

    def _key_jump_frames(self, event):
        """Handler for left/right arrow keys to jump frames. Shift key is for large jumps."""
        # Don't hijack arrow keys inside text/entry/combobox widgets (let cursor move)
        try:
            wclass = event.widget.winfo_class() if event else ""
        except Exception:
            wclass = ""
        if wclass in ("TEntry", "Entry", "TCombobox", "Combobox", "TSpinbox", "Spinbox", "Text"):
            return
        if not self.source_readers:
            return

        # Determine jump size: 1 for normal, 10 for Shift (state mask 0x1)
        jump_size = 1
        if event.state & 0x1:  # 0x1 is the mask for the Shift key state
            jump_size = 10

        direction_multiplier = 0
        if event.keysym == "Left":
            direction_multiplier = -1
        elif event.keysym == "Right":
            direction_multiplier = 1
        elif event.keysym == "Shift_L" or event.keysym == "Shift_R":
            # Ignore just the Shift keypress itself if it somehow triggered the event
            return
        else:
            return  # Should not happen

        self._jump_frames_by(direction_multiplier * jump_size)

    def _key_toggle_play_pause(self, event=None):
        """Spacebar: Play/Pause (frame-by-frame)."""
        # Don't hijack inside text/entry/combobox widgets (let them behave normally)
        try:
            wclass = event.widget.winfo_class() if event else ""
        except Exception:
            wclass = ""
        if wclass in ("TEntry", "Entry", "TCombobox", "Combobox", "TSpinbox", "Spinbox", "Text"):
            return

        self._toggle_play_pause()
        return "break"

    def _key_shift_space_fast_forward(self, event=None):
        """Shift+Spacebar: Start Fast Forward (step 5). Spacebar pauses."""
        # Don't hijack inside text/entry/combobox widgets (let them behave normally)
        try:
            wclass = event.widget.winfo_class() if event else ""
        except Exception:
            wclass = ""
        if wclass in ("TEntry", "Entry", "TCombobox", "Combobox", "TSpinbox", "Spinbox", "Text"):
            return

        if self._is_playing and self._play_step == 5:
            self._stop_playback()
        else:
            self._start_playback(step=5)
        return "break"

    def _toggle_play_pause(self):
        """Button/Spacebar handler: toggles 1-frame playback."""
        if self._is_playing:
            self._stop_playback()
        else:
            self._start_playback(step=1)

        # Keep focus off the playback buttons so Spacebar shortcuts don't double-trigger.
        try:
            self.preview_canvas.focus_set()
        except Exception:
            pass



    def _toggle_fast_forward(self):
        """Button handler: toggles fast-forward playback (step 5)."""
        if self._is_playing and self._play_step == 5:
            self._stop_playback()
        else:
            self._start_playback(step=5)

        # Keep focus off the playback buttons so Spacebar shortcuts don't double-trigger.
        try:
            self.preview_canvas.focus_set()
        except Exception:
            pass



    def _start_playback(self, step: int = 1):
        """Starts preview playback using the existing frame-advance pathway."""
        if not self.source_readers:
            return

        # Ensure any prior scheduled tick is cancelled before starting.
        if self._is_playing:
            self._stop_playback()

        self._is_playing = True
        self._play_step = 5 if int(step) == 5 else 1
        self._update_playback_button_labels()
        self._update_loop_indicator()
        self._schedule_playback_tick()

    def _stop_playback(self):
        """Stops preview playback and cancels any scheduled tick."""
        self._is_playing = False
        if getattr(self, "_play_after_id", None):
            try:
                self.root_window.after_cancel(self._play_after_id)
            except Exception:
                pass
        self._play_after_id = None
        self._update_playback_button_labels()
        self._update_loop_indicator()

    def _schedule_playback_tick(self):
        """Schedules the next playback tick (yields to Tk event loop)."""
        if not self._is_playing:
            return

        if getattr(self, "_play_after_id", None):
            try:
                self.root_window.after_cancel(self._play_after_id)
            except Exception:
                pass
            self._play_after_id = None

        # Use after(0) to yield to Tk's event loop (keeps UI responsive during playback).
        self._play_after_id = self.root_window.after(0, self._playback_tick)

    def _playback_tick(self):
        """Advances frames repeatedly until paused or reaching the end."""
        self._play_after_id = None

        if not self._is_playing:
            return
        if not self.source_readers:
            self._stop_playback()
            return

        total_frames = int(self.frame_scrubber.cget("to")) + 1
        if total_frames <= 0:
            self._stop_playback()
            return

        current_frame = int(self.frame_scrubber_var.get())
        if current_frame >= total_frames - 1:
            if getattr(self, "loop_playback_var", None) is not None and bool(self.loop_playback_var.get()) and total_frames > 1:
                self.frame_scrubber_var.set(0)
                self.on_scrubber_move(0)
                self.update_preview()
                self._schedule_playback_tick()
                return
            self._stop_playback()
            return

        prev_frame = current_frame
        self._jump_frames_by(self._play_step)
        new_frame = int(self.frame_scrubber_var.get())

        # Stop when we hit the end (or couldn't advance).
        if new_frame >= total_frames - 1 or new_frame == prev_frame:
            if getattr(self, "loop_playback_var", None) is not None and bool(self.loop_playback_var.get()) and total_frames > 1:
                self.frame_scrubber_var.set(0)
                self.on_scrubber_move(0)
                self.update_preview()
                self._schedule_playback_tick()
                return
            self._stop_playback()
            return

        self._schedule_playback_tick()

    def _update_playback_button_labels(self):
        """Updates the Play/Pause button label. Safe to call even before widgets exist."""
        if hasattr(self, "play_pause_button"):
            try:
                self.play_pause_button.config(text="⏸" if self._is_playing else "▶")
            except Exception:
                pass
    def _toggle_loop(self, event=None):
        """Toggle loop playback on/off (UI-only state)."""
        try:
            self.loop_playback_var.set(not bool(self.loop_playback_var.get()))
        except Exception:
            # If somehow not initialized, default to enabling
            try:
                self.loop_playback_var = tk.BooleanVar(value=True)
            except Exception:
                return
        self._update_loop_indicator()

    def _update_loop_indicator(self):
        """Update the loop indicator appearance (latched + dim when stopped)."""
        if not hasattr(self, "loop_label"):
            return
        try:
            loop_on = bool(self.loop_playback_var.get())
        except Exception:
            loop_on = False

        # Dim when playback is stopped; brighter while actively playing.
        if loop_on:
            fg = "#2f5fb8" if self._is_playing else "#5f5f5f"  # dark blue / dark gray
            relief = "sunken"
            bd = 1
            padx = 3
            pady = 0
        else:
            fg = "#5f5f5f" if self._is_playing else "#9a9a9a"  # dark gray / gray
            relief = "flat"
            bd = 0
            padx = 0
            pady = 0

        try:
            self.loop_label.config(fg=fg, relief=relief, bd=bd, padx=padx, pady=pady)
        except Exception:
            pass


    def load_video_list(self, find_sources_callback: Callable):
        """
        Loads a list of video sources to be previewed.

        Args:
            find_sources_callback (Callable): A function that returns a list of dictionaries.
        """
        self.video_list = find_sources_callback()

        if not self.video_list:
            messagebox.showwarning("Not Found", "No valid source videos found.")
            self.current_video_index = -1
            self._update_nav_controls()
            return

        target_index = 0
        
        if self.last_loaded_video_path:
            # Search for the index matching the last loaded path
            for i, source_dict in enumerate(self.video_list):
                if source_dict.get('source_video') == self.last_loaded_video_path:
                    target_index = i
                    logger.debug(f"Last loaded video path found at new index: {target_index}")
                    break
            else:
                # Path not found (e.g., file was removed/renamed)
                self.last_loaded_frame_index = 0 # Reset frame scrubber
                logger.debug("Last loaded video path NOT found in new list. Resetting to index 0.")


        self.current_video_index = target_index # Use the recalled or default index
        self._load_preview_by_index(self.current_video_index)

    def _load_preview_by_index(self, index: int):
        """Loads a specific video from the preview list by its index."""
        self._clear_preview_resources()

        if not (0 <= index < len(self.video_list)):
            self.last_loaded_video_path = None
            return

        self.current_video_index = index
        self._update_nav_controls()

        source_paths = self.video_list[index]
        base_name = os.path.basename(next(iter(source_paths.values())))
        
        main_source_path = source_paths.get('source_video', None)

        initial_frame = 0
        if main_source_path == self.last_loaded_video_path:
            # If the path is the SAME, retain the last frame index.
            initial_frame = self.last_loaded_frame_index
        else:
            # If the path is DIFFERENT (new video), reset frame index to 0.
            self.last_loaded_frame_index = 0
            
        self.last_loaded_video_path = main_source_path

        self.load_preview_button.config(text="LOADING...", style="Loading.TButton")
        self.parent.update_idletasks()

        try:
            # Initialize VideoReader for each source path
            num_frames = -1
            for key, path in source_paths.items():
                # --- MODIFIED: Explicitly check for valid path and skip if None/empty ---
                if not isinstance(path, str) or not path or not os.path.exists(path):
                    self.source_readers[key] = None
                    # Log only if the key is expected to have a path (i.e., not a flag like 'is_sbs_input')
                    if key not in ['is_sbs_input', 'is_quad_input']:
                         logger.debug(f"Source '{key}' skipped. Path is not a string or file not found: {path}")
                    continue
                # --- END MODIFIED ---
                
                try:
                    reader = VideoReader(path, ctx=cpu(0))
                    if num_frames == -1:
                        num_frames = len(reader)
                    elif num_frames != len(reader):
                        raise ValueError(f"Frame count mismatch between sources for {base_name}")
                    self.source_readers[key] = reader
                except Exception as e:
                    self.source_readers[key] = None
                    logger.error(f"Failed to open reader for source '{key}' at path '{path}': {e}", exc_info=True)
                    # If the main sources fail, we should stop trying to load
                    if key in ['inpainted', 'splatted']:
                         raise ValueError(f"Critical source file '{key}' failed to load: {e}")

            # Depth-map stream probe (bit depth) - cached per clip
            self._depth_path = source_paths.get('depth_map') if isinstance(source_paths, dict) else None
            self._depth_msb_shift = None
            self._depth_bit_depth = 8
            self._depth_is_high_bit = False
            self._depth_native_w = None
            self._depth_native_h = None

            if self._depth_path and os.path.exists(self._depth_path):
                try:
                    depth_info = get_video_stream_info(self._depth_path)
                    pix = str((depth_info or {}).get("pix_fmt", "")).lower()
                    profile = str((depth_info or {}).get("profile", "")).lower()

                    # Conservative, best-effort inference
                    if "p16" in pix or "16" in pix or pix.startswith("gray16"):
                        self._depth_bit_depth = 16
                    elif "p12" in pix or "12" in pix:
                        self._depth_bit_depth = 12
                    elif "p10" in pix or "10" in pix or "main10" in profile:
                        self._depth_bit_depth = 10
                    else:
                        self._depth_bit_depth = 8

                    self._depth_is_high_bit = self._depth_bit_depth > 8
                except Exception as e:
                    logger.warning(f"Previewer: depth bit-depth probe failed for '{self._depth_path}': {e}")

            # Cache native depth size for ffmpeg single-frame decode
            depth_reader = self.source_readers.get("depth_map")
            if depth_reader:
                try:
                    _df0 = depth_reader.get_batch([0]).asnumpy()
                    self._depth_native_h, self._depth_native_w = _df0.shape[1:3]
                except Exception:
                    pass


            # Configure the scrubber
            self.frame_scrubber.config(to=num_frames - 1)
            initial_frame = min(initial_frame, num_frames - 1) 
            self.frame_scrubber_var.set(initial_frame)
            self.on_scrubber_move(initial_frame)
            if self.update_clip_callback:
                self.update_clip_callback()
            
            if self.parent and hasattr(self.parent, 'update_gui_from_sidecar'):
                self.parent.update_gui_from_sidecar(source_paths.get('depth_map'))

            self.update_preview()

        except Exception as e:
            messagebox.showerror("Preview Load Error", f"Failed to load files for preview:\n\n{e}")
            logger.error("Preview load failed", exc_info=True)
        finally:
            self.load_preview_button.config(text="Load/Refresh List", style="TButton")

    def _nav_preview_video(self, direction: int):
        """Navigate to the previous or next video in the preview list.

        Coalesces rapid presses (buttons/keys) so skipping over many clips only triggers one load,
        but updates the UI counter immediately so it doesn't feel like nothing happened.
        """
        try:
            if not self.video_list:
                return

            # Base index is the index at the start of a nav burst (before the delayed load happens)
            after_id = getattr(self, "_nav_pending_after_id", None)
            if after_id is None:
                self._nav_base_index = int(getattr(self, "current_video_index", 0))

            # Accumulate delta during the burst
            self._nav_pending_delta = int(getattr(self, "_nav_pending_delta", 0)) + int(direction)

            # Compute the *pending* target index for immediate UI feedback (no load yet)
            total_videos = len(self.video_list)
            base_index = int(getattr(self, "_nav_base_index", getattr(self, "current_video_index", 0)))
            pending_target = base_index + int(getattr(self, "_nav_pending_delta", 0))
            if pending_target < 0:
                pending_target = 0
            elif pending_target >= total_videos:
                pending_target = total_videos - 1

            try:
                self.video_status_label_var.set(f"Video: {pending_target + 1} / {total_videos}" if total_videos > 0 else "Video: 0 / 0")
                self.video_jump_to_var.set(str(pending_target + 1) if total_videos > 0 else "1")
                # Update button enabled state based on pending target (so it feels responsive)
                self.prev_video_button.config(state="normal" if pending_target > 0 else "disabled")
                self.next_video_button.config(state="normal" if pending_target < total_videos - 1 else "disabled")
            except Exception:
                pass

            # Cancel any queued nav flush so we only execute once after user stops spamming input
            if after_id is not None:
                try:
                    self.root_window.after_cancel(after_id)
                except Exception:
                    pass

            # Flush a bit after the last press
            self._nav_pending_after_id = self.root_window.after(120, self._nav_preview_video_flush)
            return
        except Exception:
            # If anything goes wrong, fall back to immediate nav
            pass

        self._nav_preview_video_apply(direction)

    def _nav_preview_video_flush(self):
        try:
            delta = int(getattr(self, "_nav_pending_delta", 0))
        except Exception:
            delta = 0
        try:
            base_index = int(getattr(self, "_nav_base_index", getattr(self, "current_video_index", 0)))
        except Exception:
            base_index = int(getattr(self, "current_video_index", 0))

        # Clear pending state
        try:
            self._nav_pending_delta = 0
            self._nav_pending_after_id = None
            self._nav_base_index = None
        except Exception:
            pass

        if delta == 0:
            return

        self._nav_preview_video_apply(base_index + delta, absolute=True)

    def _nav_preview_video_apply(self, direction: int, absolute: bool = False):
        """Apply the actual navigation immediately (single load).

        If absolute=True, `direction` is treated as the target index.
        """
        self._stop_playback()
        if not self.video_list:
            return

        # --- Auto-Save Current Sidecar before navigating ---
        if self.on_clip_navigate_callback:
            self.on_clip_navigate_callback()

        if absolute:
            new_index = int(direction)
        else:
            new_index = self.current_video_index + int(direction)

        if new_index < 0:
            new_index = 0
        elif new_index >= len(self.video_list):
            new_index = len(self.video_list) - 1

        if 0 <= new_index < len(self.video_list):
            self._load_preview_by_index(new_index)

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
        self.last_loaded_frame_index = frame_idx

    def _on_scrubber_trough_click(self, event):
        """Handles clicks on the frame scrubber's trough for precise positioning."""
        slider = self.frame_scrubber
        # Check if the click is on the trough to avoid interfering with handle drags
        if 'trough' in slider.identify(event.x, event.y):
            # Force the widget to update its size info to get an accurate width
            slider.update_idletasks()
            from_ = slider.cget("from")
            to = slider.cget("to")
            
            new_value = from_ + (to - from_) * (event.x / slider.winfo_width())
            self.frame_scrubber_var.set(new_value) # This triggers on_scrubber_move
            self.on_scrubber_move(new_value)
            self.on_slider_release(event) # Manually trigger preview update
            
            return "break" # Prevents the default slider click behavior

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
                # self.pil_image_for_preview already holds the correctly scaled image from update_preview
                self.pil_image_for_preview.save(filepath)
                logger.info(f"Preview frame saved to: {filepath}")
            except Exception as e:
                logger.error(f"Failed to save preview frame: {e}", exc_info=True)
                messagebox.showerror("Save Error", f"An error occurred while saving the image:\n{e}")

    def set_parameters(self, params: Dict[str, Any]):
        """
        Receives a dictionary of parameters from the main GUI.
        Triggers a preview update if the parameters have changed.
        """
        # This method is now primarily for external triggers.
        # The main way of getting params is now the get_params_callback.
        self.update_preview()

    def set_preview_source_options(self, options: list):
        """Sets the available options for the preview source dropdown."""
        current_val = self.preview_source_combo.get()
        self.preview_source_combo['values'] = options
        if current_val in options:
            self.preview_source_combo.set(current_val)
        elif options:
            self.preview_source_combo.set(options[0])

    def set_ui_processing_state(self, is_processing: bool):
        """
        Disables or enables all interactive widgets in the previewer during batch processing.
        """
        state = "disabled" if is_processing else "normal"
        for widget in self.widgets_to_disable:
            try:
                # Special handling for combobox which uses 'readonly' instead of 'normal'
                if isinstance(widget, ttk.Combobox):
                    widget.config(state="disabled" if is_processing else "readonly")
                else:
                    widget.config(state=state)
            except tk.TclError:
                pass # Ignore if widgets don't exist yet
            
    def _start_wigglegram_animation(self, left_frame: torch.Tensor, right_frame: torch.Tensor):
        """Starts the wigglegram animation loop."""
        self._stop_wigglegram_animation()

        # --- MODIFIED: Use percentage scaling for wigglegram frames ---
        scale_percent_str = self.preview_size_var.get()
        try:
            scale_factor = float(scale_percent_str.strip('%')) / 100.0
        except ValueError:
            scale_factor = 1.0

        def scale_image_for_wiggle(frame_tensor: torch.Tensor) -> ImageTk.PhotoImage:
            """Scales a single frame tensor to a PhotoImage using the calculated factor."""
            frame_np = (frame_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(frame_np)
            
            if scale_factor != 1.0 and scale_factor > 0:
                new_width = int(pil_img.width * scale_factor)
                new_height = int(pil_img.height * scale_factor)
                if new_width > 0 and new_height > 0:
                    pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return ImageTk.PhotoImage(pil_img)

        self.wiggle_left_tk = scale_image_for_wiggle(left_frame)
        self.wiggle_right_tk = scale_image_for_wiggle(right_frame)
        # --- END MODIFIED ---

        self._wiggle_step(True)

    def _stop_wigglegram_animation(self):
        if self.wiggle_after_id:
            self.parent.after_cancel(self.wiggle_after_id)
            self.wiggle_after_id = None
        if hasattr(self, 'wiggle_left_tk'): del self.wiggle_left_tk
        if hasattr(self, 'wiggle_right_tk'): del self.wiggle_right_tk
    
    def _update_nav_controls(self):
        """Updates the state and labels of the video navigation controls."""
        total_videos = len(self.video_list)
        current_index = self.current_video_index

        self.video_status_label_var.set(f"Video: {current_index + 1} / {total_videos}" if total_videos > 0 else "Video: 0 / 0")
        self.video_jump_to_var.set(str(current_index + 1) if total_videos > 0 else "1")

        self.prev_video_button.config(state="normal" if current_index > 0 else "disabled")
        self.next_video_button.config(state="normal" if 0 <= current_index < total_videos - 1 else "disabled")
        self.video_jump_entry.config(state="normal" if total_videos > 0 else "disabled")

    def update_preview(self):
        """The main preview generation function."""
        if not self.source_readers:
            return

        # --- NEW: Get fresh parameters via callback ---
        if self.get_params_callback:
            self.current_params = self.get_params_callback()
        else:
            logger.warning("Previewer: get_params_callback not provided. Using stale parameters.")

        self._stop_wigglegram_animation()
        self.load_preview_button.config(text="LOADING...", style="Loading.TButton")
        self.parent.update_idletasks()

        try:
            frame_idx = int(self.frame_scrubber_var.get())

            # Load the single frame from each source reader
            source_frames = {}
            for key, reader in self.source_readers.items():
                if reader:
                    if key == 'depth_map':
                        if self._depth_is_high_bit and self._depth_path and self._depth_native_w and self._depth_native_h:
                            frame_np = self._read_depth_frame_ffmpeg(frame_idx)
                        else:
                            frame_np = reader.get_batch([frame_idx]).asnumpy()
                        # IMPORTANT: keep depth as RAW values (8-bit stays 0..255, 10-bit stays 0..1023+)
                        frame_tensor = torch.from_numpy(frame_np).permute(0, 3, 1, 2).float()
                    else:
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

            # --- MODIFIED: Calculate scale factor from percentage string and apply resizing ---
            scale_percent_str = self.current_params.get("preview_size", "100%")
            display_image = self.pil_image_for_preview.copy()

            try:
                scale_factor = float(scale_percent_str.strip('%')) / 100.0
            except ValueError:
                scale_factor = 1.0
                logger.warning(f"Invalid preview scale '{scale_percent_str}', defaulting to 100%.")

            if scale_factor != 1.0 and scale_factor > 0:
                new_width = int(display_image.width * scale_factor)
                new_height = int(display_image.height * scale_factor)
                
                # Use Image.resize to handle both scaling up and scaling down
                display_image = display_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logger.debug(f"Preview scaled by {scale_percent_str} to {new_width}x{new_height}.")
            # --- END MODIFIED ---

                        # Preview-only crosshair/bullseye overlay (drawn onto the preview image so it always shows)
            if getattr(self, "crosshair_enabled", False) and self._crosshair_allowed_for_current_source():
                try:
                    draw = ImageDraw.Draw(display_image)
                    w_img, h_img = display_image.size
                    cx, cy = w_img // 2, h_img // 2
                    color = (255, 255, 255) if getattr(self, "crosshair_white", False) else (0, 0, 0)

                    def _draw_bullseye(x: int, y: int, half_len: int, r: int, line_w: int):
                        # cross
                        draw.line((x - half_len, y, x + half_len, y), fill=color, width=line_w)
                        draw.line((x, y - half_len, x, y + half_len), fill=color, width=line_w)
                        # ring
                        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=max(1, line_w))

                    def _draw_dot(x: int, y: int, r: int):
                        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=color)

                    # Center bullseye (slightly larger)
                    base = max(8, min(50, int(min(w_img, h_img) * 0.04)))
                    _draw_bullseye(cx, cy, half_len=base, r=max(4, base // 3), line_w=3)

                    if getattr(self, "crosshair_multi", False):
                        # Outer targets: halfway between screen edge and center -> rectangle around center.
                        dx = w_img // 4
                        dy = h_img // 4

                        # Make outer targets a bit smaller than center
                        outer_len = max(6, int(base * 0.65))
                        outer_r = max(3, outer_len // 3)
                        outer_w = 2

                        # 8 outer target positions (4 midpoints + 4 corners)
                        outer = [
                            (cx - dx, cy), (cx + dx, cy), (cx, cy - dy), (cx, cy + dy),
                            (cx - dx, cy - dy), (cx + dx, cy - dy), (cx - dx, cy + dy), (cx + dx, cy + dy),
                        ]
                        for (x, y) in outer:
                            _draw_bullseye(x, y, half_len=outer_len, r=outer_r, line_w=outer_w)

                        # Dots between center and each outer target (2 per line)
                        dot_r = 2
                        for (x, y) in outer:
                            for t in (1/3, 2/3):
                                mx = int(round(cx + (x - cx) * t))
                                my = int(round(cy + (y - cy) * t))
                                _draw_dot(mx, my, dot_r)

                        # Dots along the rectangle path between outer targets (2 per segment)
                        # Order around the perimeter (clockwise), using 8 points (midpoints + corners)
                        ring = [
                            (cx, cy - dy), (cx + dx, cy - dy), (cx + dx, cy), (cx + dx, cy + dy),
                            (cx, cy + dy), (cx - dx, cy + dy), (cx - dx, cy), (cx - dx, cy - dy),
                        ]
                        for i in range(len(ring)):
                            x0, y0 = ring[i]
                            x1, y1 = ring[(i + 1) % len(ring)]
                            for t in (1/3, 2/3):
                                mx = int(round(x0 + (x1 - x0) * t))
                                my = int(round(y0 + (y1 - y0) * t))
                                _draw_dot(mx, my, dot_r)


                except Exception:
                    pass
            # Depth/Pop separation readout (preview-only; percent of screen width)
            # Independent of the crosshair toggle; color follows the White checkbox.
            if getattr(self, "depth_pop_enabled", False) and self._crosshair_allowed_for_current_source():
                try:
                    d_pct = getattr(self, "depth_pop_depth_pct", None)
                    p_pct = getattr(self, "depth_pop_pop_pct", None)
                    if d_pct is not None and p_pct is not None:
                        draw = ImageDraw.Draw(display_image)
                        w_img, h_img = display_image.size
                        cx = w_img // 2
                        color = (255, 255, 255) if getattr(self, "crosshair_white", False) else (0, 0, 0)

                        # Depth/Pop readout + Total (Total = behind + out)
                        total_pct = float(d_pct) + float(p_pct)
                        # Per-frame Total (D+P) shown in parentheses; purely informational (not saved)
                        txt = f"{float(d_pct):.1f}/{float(p_pct):.1f}% ({total_pct:.1f})"

                        # Per-clip running max (updates as frames are previewed)
                        max_total = getattr(self, "_dp_total_max_seen", None)
                        if max_total is None:
                            max_total = total_pct
                        max_txt = f"Max:{float(max_total):.1f}%"

                        # Slightly larger font (fallbacks safely if TTF isn't available)
                        try:
                            font_size = max(10, min(20, int(h_img * 0.025)))
                            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
                        except Exception:
                            try:
                                font_size = max(10, min(20, int(h_img * 0.025)))
                                font = ImageFont.truetype("arial.ttf", font_size)
                            except Exception:
                                font = ImageFont.load_default()

                        try:
                            bbox = draw.textbbox((0, 0), txt, font=font)
                            tw, th = (bbox[2] - bbox[0]), (bbox[3] - bbox[1])
                        except Exception:
                            tw, th = draw.textsize(txt, font=font)

                        x_txt = cx - (tw // 2)
                        y_txt = h_img - th - 10
                        draw.text((x_txt, y_txt), txt, fill=color, font=font)

                        # Draw max total at bottom-right (same baseline)
                        try:
                            bbox2 = draw.textbbox((0, 0), max_txt, font=font)
                            tw2, th2 = (bbox2[2] - bbox2[0]), (bbox2[3] - bbox2[1])
                        except Exception:
                            tw2, th2 = draw.textsize(max_txt, font=font)
                        x2 = w_img - tw2 - 10
                        draw.text((x2, y_txt), max_txt, fill=color, font=font)
                except Exception:
                    pass

            self.preview_image_tk = ImageTk.PhotoImage(display_image)
            # --- FIX: Attach the image reference to the widget to prevent garbage collection ---
            self.preview_label.config(image=self.preview_image_tk, text="")
            self.preview_label.image = self.preview_image_tk
            # --- END FIX ---

            # --- NEW: Trigger parent window resize ---
            # if self.resize_callback:
            #     # Force the parent to update its layout to see the new image size
            #     self.parent.update_idletasks()
            #     self.resize_callback()
            # --- END NEW ---
            self._update_preview_layout()

        except Exception as e:
            logger.error(f"Error updating preview: {e}", exc_info=True)
            self.preview_label.config(image=None, text=f"Error:\n{e}")
        finally:
            release_cuda_memory()
            self.load_preview_button.config(text="Load/Refresh List", style="TButton")

    def _read_depth_frame_ffmpeg(self, frame_idx: int) -> np.ndarray:
        """Decode a single depth frame preserving 10-bit+ using ffmpeg (fast seek)."""
        depth_reader = self.source_readers.get("depth_map")
        if not depth_reader or not self._depth_path or not self._depth_native_w or not self._depth_native_h:
            raise RuntimeError("Depth reader/path/size not initialized for ffmpeg decode.")

        # Prefer depth reader FPS (should match source), fallback to source_video FPS.
        fps = 0.0
        try:
            fps = float(depth_reader.get_avg_fps())
        except Exception:
            try:
                src = self.source_readers.get("source_video")
                fps = float(src.get_avg_fps()) if src else 0.0
            except Exception:
                fps = 0.0

        t = (float(frame_idx) / fps) if fps and fps > 0 else 0.0
        w, h = int(self._depth_native_w), int(self._depth_native_h)
        expected_bytes = w * h * 2  # gray16le

        def _run(vf: str) -> bytes:
            cmd = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel", "error",
                "-nostdin",
                "-ss", f"{t:.6f}",
                "-i", self._depth_path,
                "-an", "-sn", "-dn",
                "-frames:v", "1",
                "-vf", vf,
                "-f", "rawvideo",
                "pipe:1",
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            out = b""
            try:
                if p.stdout:
                    out = p.stdout.read(expected_bytes)
            finally:
                try:
                    p.terminate()
                except Exception:
                    pass
            return out

        # First try extracting luma plane (best for yuv depth encodes)
        buf = _run("extractplanes=y,format=gray16le")
        if len(buf) != expected_bytes:
            # Fallback for true gray sources (extractplanes may not apply)
            buf = _run("format=gray16le")

        if len(buf) != expected_bytes:
            logger.warning(f"Previewer: ffmpeg depth decode returned {len(buf)} bytes (expected {expected_bytes}); falling back to Decord 8-bit.")
            return depth_reader.get_batch([frame_idx]).asnumpy()

        arr = np.frombuffer(buf, dtype=np.uint16).reshape(1, h, w, 1)

        # Detect MSB-aligned samples (common when decoding 10-bit into 16-bit containers)
        if self._depth_msb_shift is None:
            bd = int(self._depth_bit_depth) if self._depth_bit_depth else 16
            if 0 < bd < 16:
                expected_max = (1 << bd) - 1
                max_val = int(arr.max(initial=0))
                if max_val > expected_max:
                    shift = 16 - bd
                    if shift > 0 and (max_val % (1 << shift) == 0):
                        self._depth_msb_shift = shift
                    else:
                        self._depth_msb_shift = 0
                else:
                    self._depth_msb_shift = 0
            else:
                self._depth_msb_shift = 0

        if self._depth_msb_shift:
            arr = (arr >> self._depth_msb_shift).astype(np.uint16, copy=False)

        return arr.copy()

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
        # REPLACED VERSION: Force scrollregion to encompass full image boundaries
        self.preview_canvas.config(scrollregion=(0, 0, max(canvas_w, img_w), max(canvas_h, img_h)))

    def replace_source_path_for_current_video(self, key: str, path: str):
        """Replace the source path for the currently loaded video for `key` (e.g., 'depth_map').

        This closes the old reader (if any), opens a new VideoReader for the given path (if it exists),
        updates internal caches used by the preview pipeline, and triggers an immediate preview update.
        """
        # Close previous reader if present
        try:
            old_reader = self.source_readers.get(key)
            if old_reader is not None:
                try:
                    del old_reader
                except Exception:
                    pass
                self.source_readers[key] = None

            # Attempt to open new reader if path is a valid file
            opened_reader = None
            if isinstance(path, str) and path and os.path.exists(path):
                try:
                    opened_reader = VideoReader(path, ctx=cpu(0))
                    self.source_readers[key] = opened_reader
                except Exception as e:
                    logger.error(f"replace_source_path_for_current_video: failed to open {path}: {e}")
                    self.source_readers[key] = None
            else:
                # Path invalid or missing - set None
                self.source_readers[key] = None

            # IMPORTANT (10-bit preview path):
            # update cached depth-map probe fields so map switches actually take effect.
            if key == "depth_map":
                self._depth_path = path if (isinstance(path, str) and path and os.path.exists(path)) else None
                self._depth_msb_shift = None
                self._depth_bit_depth = 8
                self._depth_is_high_bit = False
                self._depth_native_w = None
                self._depth_native_h = None

                if self._depth_path:
                    try:
                        depth_info = get_video_stream_info(self._depth_path)
                        pix = str((depth_info or {}).get("pix_fmt", "")).lower()
                        profile = str((depth_info or {}).get("profile", "")).lower()

                        if "p16" in pix or "16" in pix or pix.startswith("gray16"):
                            self._depth_bit_depth = 16
                        elif "p12" in pix or "12" in pix:
                            self._depth_bit_depth = 12
                        elif "p10" in pix or "10" in pix or "main10" in profile:
                            self._depth_bit_depth = 10
                        else:
                            self._depth_bit_depth = 8

                        self._depth_is_high_bit = self._depth_bit_depth > 8
                    except Exception as e:
                        logger.warning(f"Previewer: depth bit-depth probe failed for '{self._depth_path}': {e}")

                    # Cache native depth size for ffmpeg single-frame decode
                    try:
                        rdr = opened_reader or self.source_readers.get("depth_map")
                        if rdr is not None:
                            _df0 = rdr.get_batch([0]).asnumpy()
                            self._depth_native_h, self._depth_native_w = _df0.shape[1:3]
                    except Exception:
                        pass

        except Exception as e:
            logger.exception(f"Error replacing source reader for key '{key}': {e}")

        # Force an immediate preview refresh using the new reader
        try:
            self.update_preview()
        except Exception as e:
            logger.exception(f"Error updating preview after replacing source path: {e}")

    def _wiggle_step(self, show_left: bool):
        """A single step in the wigglegram animation."""
        if not hasattr(self, 'wiggle_left_tk'): return # Stop if resources were cleared
        current_image = self.wiggle_left_tk if show_left else self.wiggle_right_tk
        self.preview_label.config(image=current_image)
        self.preview_label.image = current_image # Prevent garbage collection
        self.wiggle_after_id = self.parent.after(60, self._wiggle_step, not show_left)
    

    def _on_preview_vscroll(self, *args):
        """Vertical scrollbar handler."""
        self.preview_canvas.yview(*args)

    def _on_preview_hscroll(self, *args):
        """Horizontal scrollbar handler."""
        self.preview_canvas.xview(*args)

    
    def _crosshair_allowed_for_current_source(self) -> bool:
        """
        Crosshair is only meaningful for anaglyph preview modes.
        Return True when the current preview source is one of the anaglyph modes.
        """
        try:
            src = self.preview_source_combo.get()
        except Exception:
            src = ""
        return src in ("Anaglyph 3D", "Dubois Anaglyph", "Optimized Anaglyph")

# --- Crosshair overlay (preview only) ---
    def set_crosshair_settings(self, enabled: bool, white: bool = False, multi: bool = False):
        """Enable/disable a center crosshair overlay. Preview-only (never exported)."""
        self.crosshair_enabled = bool(enabled)
        self.crosshair_white = bool(white)
        self.crosshair_multi = bool(multi)
        # Redraw the current preview image so the crosshair appears/disappears immediately
        try:
            self.update_preview()
        except Exception:
            pass



    def set_depth_pop_enabled(self, enabled: bool):
        """Enable/disable the Depth/Pop readout overlay. Preview-only (never exported)."""
        self.depth_pop_enabled = bool(enabled)
        # Reset running max when toggled
        if not self.depth_pop_enabled:
            self._dp_total_max_seen = None
            self._dp_total_max_video_index = None
            self._dp_signature = None
        else:
            self._dp_total_max_seen = None
            self._dp_total_max_video_index = getattr(self, 'current_video_index', None)
        # Redraw so toggling this checkbox updates immediately
        try:
            self.update_preview()
        except Exception:
            pass

    def set_depth_pop_max_estimate(self, max_total_pct: Optional[float], signature: Optional[str] = None):
        """Set an estimated per-clip max total disparity (Total = depth + pop).

        This seeds the on-screen Max readout so you can see a good estimate without playing through.
        """
        try:
            if signature is not None:
                self._dp_est_signature = signature
            self._dp_total_max_est = float(max_total_pct) if max_total_pct is not None else None

            # Seed current running max immediately so the on-screen Max updates right away.
            # If the preview hasn't produced a signature yet, adopt this one.
            current_sig = getattr(self, "_dp_signature", None)
            if signature is not None and current_sig != signature:
                # If settings changed but a new preview frame hasn't been processed yet,
                # let this estimate take over (and avoid mixing max values across different settings).
                self._dp_signature = signature
                current_sig = signature
                self._dp_total_max_seen = None

            if signature is None or signature == current_sig:
                self._dp_total_max_seen = self._dp_total_max_est
                self._dp_total_max_video_index = getattr(self, "current_video_index", None)
        except Exception:
            pass

        try:
            self.update_preview()
        except Exception:
            pass

    def set_depth_pop_metrics(
        self,
        depth_pct: Optional[float],
        pop_pct: Optional[float],
        signature: Optional[str] = None,
    ):
        """Store preview-only depth/pop separation metrics (percent of screen width).

        - depth_pct: positive percentage for far/background separation (screen-behind)
        - pop_pct:   positive percentage for near/foreground separation (screen-out)
        Pass None to clear.

        signature (optional): a small string that changes when the relevant parameters/map change.
        If it changes, the per-clip running max is reset.
        """
        # Reset running max if signature changes (e.g., map/disp/gamma/conv changed)
        try:
            if signature is not None and signature != getattr(self, "_dp_signature", None):
                self._dp_signature = signature

                # Drop estimate if it doesn't match this signature
                try:
                    if getattr(self, "_dp_est_signature", None) != signature:
                        self._dp_total_max_est = None
                except Exception:
                    pass

                # Seed running max from an estimate (if present) so you don't have to play through
                try:
                    est = getattr(self, "_dp_total_max_est", None)
                    self._dp_total_max_seen = float(est) if est is not None else None
                except Exception:
                    self._dp_total_max_seen = None

                self._dp_total_max_video_index = getattr(self, "current_video_index", None)
        except Exception:
            pass


        self.depth_pop_depth_pct = depth_pct
        self.depth_pop_pop_pct = pop_pct

        # Update per-clip running max (cheap; uses already-computed per-frame values)
        try:
            if self._dp_total_max_video_index != getattr(self, "current_video_index", None):
                self._dp_total_max_video_index = getattr(self, "current_video_index", None)
                try:
                    est = getattr(self, "_dp_total_max_est", None)
                    self._dp_total_max_seen = float(est) if est is not None else None
                except Exception:
                    self._dp_total_max_seen = None

            if depth_pct is not None and pop_pct is not None:
                total = float(depth_pct) + float(pop_pct)
                if self._dp_total_max_seen is None or total > self._dp_total_max_seen:
                    self._dp_total_max_seen = total
        except Exception:
            pass



            if depth_pct is not None and pop_pct is not None:
                total = float(depth_pct) + float(pop_pct)
                if self._dp_total_max_seen is None or total > self._dp_total_max_seen:
                    self._dp_total_max_seen = total
        except Exception:
            pass