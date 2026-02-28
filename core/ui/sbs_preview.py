import tkinter as tk
from tkinter import ttk
import numpy as np
import torch
from PIL import Image, ImageTk, ImageDraw, ImageFont
from typing import Optional, Tuple

class SBSPreviewWindow:
    """
    A standalone Toplevel window for Side-By-Side (SBS) stereo preview.
    Designed to be used by any GUI (Splatting, Merging, etc.) by providing
    Left and Right eye tensors.
    """
    def __init__(self, parent, title="SBS Preview", on_close_callback=None):
        self.parent = parent
        self.on_close_callback = on_close_callback
        
        self.window = tk.Toplevel(parent)
        self.window.title(title)
        self.window.configure(bg="black")
        
        # Set a reasonable initial size for 2:1 SBS (e.g., 1280x360)
        try:
            sw = self.window.winfo_screenwidth()
            sh = self.window.winfo_screenheight()
            init_w = min(1280, int(sw * 0.8))
            init_h = min(360, int(sh * 0.4))
            self.window.geometry(f"{init_w}x{init_h}")
            self.window.update_idletasks()
        except Exception:
            self.window.geometry("1280x360")

        self.window.resizable(True, True)
        
        # Use a frame to contain the label
        self.main_frame = tk.Frame(self.window, bg="black")
        self.main_frame.pack(fill="both", expand=True)
        
        self.label = tk.Label(self.main_frame, bg="black")
        self.label.pack(fill="both", expand=True)
        
        self.is_fullscreen = False
        self.is_cross_eye = False
        self.photo = None
        
        # Keybinds - Direct Controls
        self.window.bind("<F11>", lambda e: self.toggle_fullscreen())
        self.window.bind("<Escape>", lambda e: self.toggle_fullscreen(force=False))
        self.window.bind("<x>", lambda e: self.toggle_cross_eye())
        self.window.bind("<X>", lambda e: self.toggle_cross_eye())
        
        # --- NEW: Unified Key Relay ---
        # Instead of a manual list, we intercept ALL key presses
        # and promote them to the root window if they aren't for the SBS window.
        self._relaying_lock = False
        self.window.bind("<KeyPress>", self._relay_key_event)

        # Double-click toggles fullscreen on the label
        self.label.bind("<Double-Button-1>", lambda e: self.toggle_fullscreen())
        
        # Close handler
        self.window.protocol("WM_DELETE_WINDOW", self._on_close)

    def _relay_key_event(self, event):
        """Pass all key presses to the parent window for parity with the main GUI."""
        if self._relaying_lock:
            return
            
        ksy = event.keysym.lower()
        
        # 1. Handle SBS-specific shortcuts first
        if ksy == "x":
            self.toggle_cross_eye()
            return "break"
        if ksy == "f11":
            self.toggle_fullscreen()
            return "break"
        if ksy == "escape":
            if self.is_fullscreen:
                self.toggle_fullscreen(force=False)
                return "break"
            # If not fullscreen, relay Escape to let it close dialogs/menus
            
        # 2. Relay EVERYTHING else to the parent's top-level window
        if self.parent:
            root = self.parent.winfo_toplevel()
            # Construct a complete event promotion with correct flag mappings
            kwargs = {
                'serial': event.serial, 
                'time': event.time,
                'x': event.x, 
                'y': event.y, 
                'rootx': event.x_root, 
                'rooty': event.y_root,
                'state': event.state, 
                'keycode': event.keycode,
                'keysym': event.keysym
            }
            
            # Use a guard to prevent the root window from potentially
            # echoing the event back to the focused SBS window (recursion).
            try:
                self._relaying_lock = True
                # Generate a specific event for the keysym if it is a single character
                # or a known navigation key, otherwise use generic KeyPress.
                if len(event.keysym) <= 1 or event.keysym in ("Left", "Right", "Up", "Down", "space", "Prior", "Next"):
                    root.event_generate(f"<{event.keysym}>", **kwargs)
                else:
                    root.event_generate("<KeyPress>", **kwargs)
            except Exception:
                pass
            finally:
                self._relaying_lock = False
            
        return "break" # Prevent the Toplevel from doing its own handling

    def _on_close(self):
        if self.on_close_callback:
            self.on_close_callback()
        self.destroy()

    def destroy(self):
        if self.window:
            try:
                self.window.destroy()
            except Exception:
                pass
        self.window = None
        self.label = None
        self.photo = None

    def exists(self):
        return self.window is not None and self.window.winfo_exists()

    def lift(self):
        if self.exists():
            self.window.deiconify()
            self.window.lift()

    def toggle_fullscreen(self, force: Optional[bool] = None):
        if not self.exists():
            return
        if force is None:
            self.is_fullscreen = not self.is_fullscreen
        else:
            self.is_fullscreen = bool(force)
        self.window.attributes("-fullscreen", self.is_fullscreen)

    def toggle_cross_eye(self):
        self.is_cross_eye = not self.is_cross_eye
        # Request immediate update from parent if possible
        if hasattr(self.parent, "previewer"):
            self.parent.previewer.update_preview()

    def update_frame(self, left_tensor, right_tensor, overlays_callback=None):
        """
        Updates the window with new stereo frames.
        left_tensor, right_tensor: torch.Tensors [C, H, W] in [0, 1] range.
        overlays_callback: A function that takes a PIL Image and this class instance.
        """
        if not self.exists():
            return

        try:
            # Convert tensors to numpy [H, W, C] uint8
            def to_np(t):
                if t.dim() == 4: t = t.squeeze(0)
                return (t.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

            left_np = to_np(left_tensor)
            right_np = to_np(right_tensor)
            
            # Concatenate SBS (handle cross-eye swap)
            if self.is_cross_eye:
                sbs_np = np.concatenate([right_np, left_np], axis=1)
            else:
                sbs_np = np.concatenate([left_np, right_np], axis=1)
                
            img = Image.fromarray(sbs_np)
            
            # Apply external overlays (crosshairs, metrics)
            if overlays_callback:
                draw = ImageDraw.Draw(img)
                overlays_callback(img, self)
                
            # Resize logic (downscale to fit window/screen)
            w_img, h_img = img.size
            w_win = self.window.winfo_width()
            h_win = self.window.winfo_height()
            
            if w_win <= 1 or h_win <= 1:
                w_win = self.window.winfo_screenwidth()
                h_win = self.window.winfo_screenheight()

            if w_img > w_win or h_img > h_win:
                scale = min(float(w_win) / w_img, float(h_win) / h_img)
                if scale > 0:
                    img = img.resize((int(w_img * scale), int(h_img * scale)), Image.Resampling.LANCZOS)

            self.photo = ImageTk.PhotoImage(img)
            self.label.configure(image=self.photo)
            self.label.image = self.photo
        except Exception:
            pass

    @staticmethod
    def draw_bullseye_overlay(draw: ImageDraw.ImageDraw, x_off: int, half_w: int, h_img: int, color: Tuple[int, int, int], multi: bool):
        """Standardized bullseye crosshair drawing."""
        cx, cy = x_off + (half_w // 2), h_img // 2

        def _draw_bullseye(x: int, y: int, half_len: int, r: int, line_w: int):
            draw.line((x - half_len, y, x + half_len, y), fill=color, width=line_w)
            draw.line((x, y - half_len, x, y + half_len), fill=color, width=line_w)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=max(1, line_w))

        def _draw_dot(x: int, y: int, r: int):
            draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=color)

        base = max(8, min(50, int(min(half_w, h_img) * 0.04)))
        _draw_bullseye(cx, cy, half_len=base, r=max(4, base // 3), line_w=3)

        if not multi:
            return

        dx, dy = half_w // 4, h_img // 4
        outer_len = max(6, int(base * 0.65))
        outer_r, outer_w = max(3, outer_len // 3), 2

        outer = [(cx-dx, cy), (cx+dx, cy), (cx, cy-dy), (cx, cy+dy), (cx-dx, cy-dy), (cx+dx, cy-dy), (cx-dx, cy+dy), (cx+dx, cy+dy)]
        for x, y in outer:
            _draw_bullseye(x, y, half_len=outer_len, r=outer_r, line_w=outer_w)

        dot_r = 2
        for x, y in outer:
            for t in (1/3, 2/3):
                _draw_dot(int(round(cx + (x-cx)*t)), int(round(cy + (y-cy)*t)), dot_r)

        ring = [(cx, cy-dy), (cx+dx, cy-dy), (cx+dx, cy), (cx+dx, cy+dy), (cx, cy+dy), (cx-dx, cy+dy), (cx-dx, cy), (cx-dx, cy-dy)]
        for i in range(len(ring)):
            x0, y0 = ring[i]
            x1, y1 = ring[(i+1)%len(ring)]
            for t in (1/3, 2/3):
                _draw_dot(int(round(x0 + (x1-x0)*t)), int(round(y0 + (y1-y0)*t)), dot_r)
