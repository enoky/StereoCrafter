#!/usr/bin/env python3
"""
narrative_convergence_gui.py
----------------------------

Tkinter GUI for running `narrative_convergence.py` with convenient controls.

Features:
- Choose RGB videos folder and aligned depth folder (depth names may end with "_depth").
- For each processed depth map, a JSON file with the convergence value is saved in the same folder.
- Configure detector, tracker, saliency (gaze), device (CPU/CUDA), smoothing window, Kalman toggle.
- Frame stride option to process every Nth frame for speed.
- Saves all settings and paths to "config_auto_conv.json" each run and reloads them on startup.
- Live log area and determinate progress bar.
- Background thread execution to keep UI responsive.

Requirements:
    python -m pip install tkinter (Linux may need: sudo apt-get install python3-tk)
    And whatever `narrative_convergence.py` requires (opencv-python, numpy, torch, ultralytics,
    deep_sort_realtime, tqdm, scipy)
"""

from __future__ import annotations

import os
import sys
import threading
import queue
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# Import the processing implementation
try:
    from narrative_convergence import NarrativeProcessor  # type: ignore
except Exception as e:
    # Defer error dialog until a Tk root exists
    print(f"[GUI] Could not import narrative_convergence.py: {e}", file=sys.stderr)
    NarrativeProcessor = None  # type: ignore

# torch is optional; used only to decide if CUDA should appear
try:
    import torch  # type: ignore
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Narrative Convergence – GUI")
        self.geometry("980x600")
        self.minsize(980, 530)

        self._build_ui()
        self._load_settings()

        # Worker thread management
        self.worker: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()
        self.msg_q: "queue.Queue[str]" = queue.Queue()
        self.progress_total = 0
        self.progress_done = 0

        # Periodic UI updater for logs
        self.after(120, self._drain_queue)

        # Warn on import failure (now that a root exists)
        if NarrativeProcessor is None:
            messagebox.showerror("Import Error", "Could not import narrative_convergence.py.\n"
                                 "Please place it next to this GUI script and ensure it imports cleanly.")
            # still allow UI to open for path browsing

    # ----- UI Construction -----
    def _build_ui(self) -> None:
        main = ttk.Frame(self, padding=12)
        main.pack(fill="both", expand=True)

        # Paths frame
        f_paths = ttk.LabelFrame(main, text="Paths", padding=10)
        f_paths.pack(fill="x", expand=False, pady=(0, 10))

        self.var_videos = tk.StringVar()
        self.var_depths = tk.StringVar()

        self._add_path_picker(f_paths, "RGB videos folder:", self.var_videos, self._pick_dir)
        self._add_path_picker(f_paths, "Depth folder / video(s):", self.var_depths, self._pick_dir)

        # Options frame
        f_opts = ttk.LabelFrame(main, text="Options", padding=10)
        f_opts.pack(fill="x", expand=False, pady=(0, 10))

        grid = ttk.Frame(f_opts)
        grid.pack(fill="x", expand=False)

        # Detector
        ttk.Label(grid, text="Detector:").grid(row=0, column=0, sticky="w")
        self.var_detector = tk.StringVar(value="yolo12s")
        cb_detector = ttk.Combobox(grid, textvariable=self.var_detector, values=["yolo12n", "yolo12s", "yolo12m"],
                                   state="readonly", width=14)
        cb_detector.grid(row=0, column=1, padx=6, pady=4, sticky="w")

        # Tracker
        ttk.Label(grid, text="Tracker:").grid(row=0, column=2, sticky="w")
        self.var_tracker = tk.StringVar(value="deepsort")
        cb_tracker = ttk.Combobox(grid, textvariable=self.var_tracker, values=["deepsort"],
                                  state="readonly", width=14)
        cb_tracker.grid(row=0, column=3, padx=6, pady=4, sticky="w")

        # Gaze
        ttk.Label(grid, text="Gaze:").grid(row=0, column=4, sticky="w")
        self.var_gaze = tk.StringVar(value="auto")
        cb_gaze = ttk.Combobox(grid, textvariable=self.var_gaze, values=["auto", "deepvs", "pats"],
                               state="readonly", width=14)
        cb_gaze.grid(row=0, column=5, padx=6, pady=4, sticky="w")

        # Device
        ttk.Label(grid, text="Device:").grid(row=1, column=0, sticky="w")
        device_values = ["cpu"]
        if HAVE_TORCH and hasattr(torch, "cuda") and torch.cuda.is_available():  # type: ignore[attr-defined]
            device_values.append("cuda")
        self.var_device = tk.StringVar(value=("cuda" if "cuda" in device_values else "cpu"))
        cb_device = ttk.Combobox(grid, textvariable=self.var_device, values=device_values, state="readonly", width=14)
        cb_device.grid(row=1, column=1, padx=6, pady=4, sticky="w")

        # Window
        ttk.Label(grid, text="Smoothing window:").grid(row=1, column=2, sticky="w")
        self.var_window = tk.StringVar(value="9")
        sp_window = ttk.Spinbox(grid, from_=1, to=99, textvariable=self.var_window, width=12)
        sp_window.grid(row=1, column=3, padx=6, pady=4, sticky="w")

        # Frame stride
        ttk.Label(grid, text="Frame stride:").grid(row=1, column=4, sticky="w")
        self.var_stride = tk.StringVar(value="1")
        sp_stride = ttk.Spinbox(grid, from_=1, to=1000, textvariable=self.var_stride, width=12)
        sp_stride.grid(row=1, column=5, padx=6, pady=4, sticky="w")

        # Kalman
        self.var_kalman = tk.BooleanVar(value=False)
        ttk.Checkbutton(grid, text="Use Kalman filter", variable=self.var_kalman).grid(row=2, column=0, columnspan=2, sticky="w")

        # Max videos
        ttk.Label(grid, text="Max videos (optional):").grid(row=2, column=2, sticky="w")
        self.var_max = tk.StringVar(value="")
        ent_max = ttk.Entry(grid, textvariable=self.var_max, width=12)
        ent_max.grid(row=2, column=3, padx=6, pady=4, sticky="w")

        # Seed
        ttk.Label(grid, text="Seed:").grid(row=2, column=4, sticky="w")
        self.var_seed = tk.StringVar(value="42")
        ent_seed = ttk.Entry(grid, textvariable=self.var_seed, width=12)
        ent_seed.grid(row=2, column=5, padx=6, pady=4, sticky="w")

        # Verbose
        self.var_verbose = tk.BooleanVar(value=True)
        ttk.Checkbutton(grid, text="Verbose Logging", variable=self.var_verbose).grid(row=3, column=0, columnspan=2, sticky="w")

        # Debug MP4 Options
        f_dbg = ttk.LabelFrame(main, text="Debug MP4 Options", padding=10)
        f_dbg.pack(fill="x", expand=False, pady=(0, 10))

        # Save debug videos toggle
        self.var_debug_video = tk.BooleanVar(value=False)
        chk_debug = ttk.Checkbutton(f_dbg, text="Save debug MP4s", variable=self.var_debug_video)
        chk_debug.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 4))

        # Debug output folder
        ttk.Label(f_dbg, text="Debug output folder:").grid(row=1, column=0, sticky="w")
        self.var_debug_out = tk.StringVar()
        ent_dbg_out = ttk.Entry(f_dbg, textvariable=self.var_debug_out)
        ent_dbg_out.grid(row=1, column=1, padx=6, pady=2, sticky="we")
        ttk.Button(f_dbg, text="Browse…", command=lambda: self._pick_dir(self.var_debug_out)).grid(row=1, column=2, padx=(0, 0), pady=2)

        # Debug scale
        ttk.Label(f_dbg, text="Debug scale:").grid(row=2, column=0, sticky="w")
        self.var_debug_scale = tk.StringVar(value="1.0")
        ent_dbg_scale = ttk.Entry(f_dbg, textvariable=self.var_debug_scale, width=10)
        ent_dbg_scale.grid(row=2, column=1, padx=6, pady=2, sticky="w")

        # Debug layers
        ttk.Label(f_dbg, text="Debug layers (CSV):").grid(row=3, column=0, sticky="w")
        self.var_debug_layers = tk.StringVar()
        ent_dbg_layers = ttk.Entry(f_dbg, textvariable=self.var_debug_layers)
        ent_dbg_layers.grid(row=3, column=1, columnspan=3, padx=6, pady=2, sticky="we")

        # Allow f_dbg grid to expand columns properly
        for col in range(4):
            f_dbg.columnconfigure(col, weight=1)

        # Run / Cancel
        f_run = ttk.Frame(main)
        f_run.pack(fill="x", expand=False, pady=(0, 10))
        self.btn_run = ttk.Button(f_run, text="Run", command=self._on_run)
        self.btn_run.pack(side="left")
        self.btn_cancel = ttk.Button(f_run, text="Cancel", command=self._on_cancel, state="disabled")
        self.btn_cancel.pack(side="left", padx=(8, 0))

        # Progress
        f_prog = ttk.LabelFrame(main, text="Progress", padding=10)
        f_prog.pack(fill="x", expand=False, pady=(0, 10))
        self.prog = ttk.Progressbar(f_prog, orient="horizontal", mode="determinate")
        self.prog.pack(fill="x", expand=True)

        # Log
        f_log = ttk.LabelFrame(main, text="Log", padding=10)
        f_log.pack(fill="both", expand=True)
        self.txt = tk.Text(f_log, wrap="word", height=20)
        self.txt.pack(fill="both", expand=True)
        self.txt.configure(state="disabled")
        self._log("Ready.\n")

    def _add_path_picker(self, parent: tk.Widget, label: str, var: tk.StringVar, picker_cb) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", expand=False, pady=2)
        ttk.Label(row, text=label, width=22).pack(side="left")
        ent = ttk.Entry(row, textvariable=var)
        ent.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Button(row, text="Browse…", command=lambda: picker_cb(var)).pack(side="left")

    # ----- Settings Load/Save -----
    def _load_settings(self) -> None:
        """Load settings from config_auto_conv.json if it exists."""
        cfg_path = Path("config_auto_conv.json")
        if not cfg_path.exists():
            return
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            
            self.var_videos.set(cfg.get("videos", ""))
            self.var_depths.set(cfg.get("depths", ""))
            self.var_detector.set(cfg.get("detector", "yolo12s"))
            self.var_tracker.set(cfg.get("tracker", "deepsort"))
            self.var_gaze.set(cfg.get("gaze", "auto"))
            self.var_device.set(cfg.get("device", "cpu"))
            self.var_window.set(str(cfg.get("window", 9)))
            self.var_stride.set(str(cfg.get("stride", 1)))
            self.var_kalman.set(bool(cfg.get("kalman", False)))
            self.var_max.set(str(cfg.get("max_videos", "") or ""))
            self.var_seed.set(str(cfg.get("seed", 42)))
            self.var_verbose.set(bool(cfg.get("verbose", True)))

            # Debug options
            self.var_debug_video.set(bool(cfg.get("debug_video", False)))
            # Note: debug_out may be None; convert to empty string
            dbg_out = cfg.get("debug_out", "")
            self.var_debug_out.set(str(dbg_out) if dbg_out is not None else "")
            self.var_debug_scale.set(str(cfg.get("debug_scale", 1.0)))
            # For debug_layers we accept either a string or list; join if list
            dbg_layers = cfg.get("debug_layers", "")
            if isinstance(dbg_layers, list):
                self.var_debug_layers.set(",".join([str(x) for x in dbg_layers]))
            else:
                self.var_debug_layers.set(str(dbg_layers))
            
            self._log(f"Loaded settings from {cfg_path.resolve()}\n")
        except Exception as e:
            self._log(f"Warning: failed to load config: {e}\n")

    # ----- Path pickers -----
    def _pick_dir(self, var: tk.StringVar) -> None:
        d = filedialog.askdirectory(initialdir=var.get() or os.getcwd())
        if d:
            var.set(d)

    def _pick_file(self, var: tk.StringVar, title: str = "Select file") -> None:
        f = filedialog.askopenfilename(initialdir=var.get() or os.getcwd(), title=title)
        if f:
            var.set(f)

    # ----- Logging -----
    def _log(self, msg: str) -> None:
        self.txt.configure(state="normal")
        self.txt.insert("end", msg)
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _drain_queue(self) -> None:
        try:
            while True:
                msg = self.msg_q.get_nowait()
                self._log(msg)
        except queue.Empty:
            pass
        # Keep polling
        self.after(120, self._drain_queue)

    # ----- Run/Cancel -----
    def _on_run(self) -> None:
        if NarrativeProcessor is None:
            messagebox.showerror("Import Error", "narrative_convergence.py is not importable.")
            return

        # Validate inputs
        videos = Path(self.var_videos.get())
        depths = Path(self.var_depths.get())
        if not videos.exists() or not videos.is_dir():
            messagebox.showerror("Error", "Please select a valid RGB videos folder.")
            return
        if not depths.exists():
            messagebox.showerror("Error", "Please select a valid depth folder (or folder containing depth videos).")
            return

        # Validate debug options: require output folder when debug video is enabled
        if self.var_debug_video.get():
            debug_out_str = (self.var_debug_out.get() or "").strip()
            if not debug_out_str:
                messagebox.showerror("Error", "Please specify a debug output folder when 'Save debug MP4s' is enabled.")
                return
        # Save all settings/paths to JSON, including debug options
        cfg = {
            "videos": str(videos),
            "depths": str(depths),
            "detector": self.var_detector.get(),
            "tracker": self.var_tracker.get(),
            "gaze": self.var_gaze.get(),
            "device": self.var_device.get(),
            "window": int(self.var_window.get() or 9),
            "stride": int(self.var_stride.get() or 1),
            "kalman": bool(self.var_kalman.get()),
            "max_videos": (int(self.var_max.get()) if (self.var_max.get() or "").strip().isdigit() else None),
            "seed": int(self.var_seed.get() or 42),
            "verbose": bool(self.var_verbose.get()),
            # Debug options persisted to config
            "debug_video": bool(self.var_debug_video.get()),
            "debug_out": (self.var_debug_out.get() or ""),
            "debug_scale": (self.var_debug_scale.get() or "1.0"),
            "debug_layers": (self.var_debug_layers.get() or ""),
        }
        try:
            cfg_path = Path("config_auto_conv.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)
            self._log(f"Saved settings to {cfg_path.resolve()}\n")
        except Exception as e:
            self._log(f"Warning: failed to save config_auto_conv.json: {e}\n")

        # Build planned pairs to size progress
        pairs = self._match_pairs(videos, depths)
        if not pairs:
            messagebox.showwarning("No matches", "No matching video/depth basenames were found.\n"
                                  "Tip: if depth files use '*_depth' suffix, that is handled automatically.")
            return

        self.progress_total = len(pairs)
        self.progress_done = 0
        self.prog.configure(maximum=self.progress_total, value=0)

        # Disable run, enable cancel
        self.btn_run.configure(state="disabled")
        self.btn_cancel.configure(state="normal")
        self.stop_flag.clear()

        # Spawn worker thread
        self.worker = threading.Thread(
            target=self._worker_run,
            args=(videos, depths, pairs),
            daemon=True,
        )
        self.worker.start()

    def _on_cancel(self) -> None:
        if self.worker and self.worker.is_alive():
            self.stop_flag.set()
            self._log("\nCancellation requested…\n")

    # ----- Pair matching (handles '_depth' suffix on depth files) -----
    @staticmethod
    def _match_pairs(video_dir: Path, depth_dir: Path) -> List[Tuple[Path, Path]]:
        def base_from_depth_name(p: Path) -> str:
            stem = p.stem
            return stem[:-6] if stem.endswith("_depth") else stem  # strip '_depth'

        video_files = [p for p in video_dir.iterdir() if p.is_file()]
        depth_map: Dict[str, Path] = {}
        for p in depth_dir.iterdir():
            depth_map[base_from_depth_name(p)] = p

        pairs: List[Tuple[Path, Path]] = []
        for vf in sorted(video_files):
            dp = depth_map.get(vf.stem)
            if dp is not None:
                pairs.append((vf, dp))
        return pairs

    # ----- Worker thread -----
    def _worker_run(self, video_dir: Path, depth_dir: Path, pairs: List[Tuple[Path, Path]]) -> None:
        # Define logger to pipe messages from processor to the GUI queue
        gui_logger = lambda msg: self.msg_q.put(f"{msg}\n")
        
        # Dictionary to hold all results for the master JSON file
        master_convergence_list = {}

        try:
            # Prepare processor
            max_videos = self._safe_int(self.var_max.get())
            if max_videos is not None:
                pairs = pairs[:max_videos]
                self.progress_total = len(pairs)
                self.prog.configure(maximum=self.progress_total, value=0)

            # Prepare debug options
            dbg_enabled = bool(self.var_debug_video.get())
            dbg_out = None
            out_str = (self.var_debug_out.get() or "").strip()
            if out_str:
                dbg_out = Path(out_str)
                try:
                    dbg_out.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass # Fail silently if dir creation fails, handled later
            
            dbg_scale = 1.0
            try:
                dbg_scale = float(self.var_debug_scale.get() or "1.0")
            except Exception:
                pass
            
            layers_str = (self.var_debug_layers.get() or "").strip()
            dbg_layers = [s.strip().lower() for s in layers_str.split(",") if s.strip()] if layers_str else None

            processor = NarrativeProcessor(
                video_dir=video_dir,
                depth_dir=depth_dir,
                detector_name=self.var_detector.get(),
                tracker_name=self.var_tracker.get(),
                gaze_name=self.var_gaze.get(),
                use_kalman=bool(self.var_kalman.get()),
                window=int(self.var_window.get() or 9),
                device=self.var_device.get(),
                max_videos=None,  # iterate manually for progress
                seed=int(self.var_seed.get() or 42),
                verbose=bool(self.var_verbose.get()),
                stride=int(self.var_stride.get() or 1),
                logger=gui_logger,
                debug_video=dbg_enabled,
                debug_out=dbg_out,
                debug_scale=dbg_scale,
                debug_layers=dbg_layers,
            )

            for i, (rgb_path, depth_path) in enumerate(pairs, start=1):
                if self.stop_flag.is_set():
                    gui_logger("\nCancelled.")
                    break
                gui_logger(f"[{i}/{len(pairs)}] Processing: {rgb_path.name}")
                try:
                    # Per-video call for explicit progress
                    conv = processor._process_single(rgb_path, depth_path)
                    
                    # Add result to the master list
                    master_convergence_list[rgb_path.stem] = conv
                    
                    # Define and save the output JSON for this video
                    output_json_path = depth_path.with_suffix('.json')
                    output_data = {"convergence_plane": conv}
                    with open(output_json_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, indent=4)

                    gui_logger(f"  → convergence: {conv:.4f}")
                    gui_logger(f"  → Saved to: {output_json_path}")

                except Exception as e:
                    gui_logger(f"  ! Error processing {rgb_path.name}: {e}")
                
                self.progress_done += 1
                self._update_progress()

            # After the loop, save the master list if a debug path was set and we have results
            if dbg_out and master_convergence_list:
                master_json_path = dbg_out / "_master_convergence_list.json"
                try:
                    with open(master_json_path, "w", encoding="utf-8") as f:
                        json.dump(master_convergence_list, f, indent=4)
                    gui_logger(f"\nSaved master convergence list to {master_json_path}")
                except Exception as e:
                    gui_logger(f"\nWarning: Failed to write master convergence list: {e}")

            if not self.stop_flag.is_set():
                gui_logger("\nProcessing complete.")

        except Exception as e:
            gui_logger(f"\nFatal error: {e}")
            messagebox.showerror("Error", str(e))
        finally:
            # Reset buttons
            self.btn_run.configure(state="normal")
            self.btn_cancel.configure(state="disabled")

    def _update_progress(self) -> None:
        # Update progress bar safely from worker
        self.prog.after(0, lambda: self.prog.configure(value=self.progress_done))

    @staticmethod
    def _safe_int(val: str) -> Optional[int]:
        val = (val or "").strip()
        if not val:
            return None
        try:
            return int(val)
        except Exception:
            return None


if __name__ == "__main__":
    App().mainloop()