import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class PreviewCanvasWindow(tk.Toplevel):
    """A Toplevel window that hosts the preview canvas, scrollbars, and image display.

    Serves as both the main preview viewport and the SBS viewer. Supports zoom,
    drag/pan, fullscreen (F11), cross-eye toggle (X key), and image centering.
    Hides on close (X button) rather than being destroyed, and can be re-shown
    via ``show()``.
    """

    def __init__(
        self,
        parent,
        title: str = "Preview",
        zoom_callback: Optional[Callable] = None,
        sbs_cross_eye_var: Optional[tk.BooleanVar] = None,
        on_close_callback: Optional[Callable] = None,
    ):
        super().__init__(parent)
        self.parent = parent
        self.title(title)
        self.configure(bg="black")

        self._zoom_callback = zoom_callback
        self.sbs_cross_eye_var = sbs_cross_eye_var
        self._on_close_callback = on_close_callback

        # State
        self._is_dragging = False
        self._is_fullscreen = False
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._pil_image: Optional[Image.Image] = None

        # Geometry
        try:
            sw = self.winfo_screenwidth()
            sh = self.winfo_screenheight()
            init_w = min(1280, int(sw * 0.8))
            init_h = min(720, int(sh * 0.7))
            self.geometry(f"{init_w}x{init_h}")
        except Exception:
            self.geometry("1280x720")

        self.resizable(True, True)

        # --- Canvas + Scrollbars ---
        self.preview_canvas = tk.Canvas(self, bg="black", highlightthickness=0)

        self.v_scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._on_vscroll)
        self.h_scrollbar = ttk.Scrollbar(self, orient="horizontal", command=self._on_hscroll)
        self.preview_canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)

        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scrollbar.grid(row=0, column=1, sticky="ns")
        self.h_scrollbar.grid(row=1, column=0, sticky="ew")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Inner frame + label for displaying the image
        self._inner_frame = ttk.Frame(self.preview_canvas)
        self._canvas_window_id = self.preview_canvas.create_window((0, 0), window=self._inner_frame, anchor="nw")
        self.preview_label = ttk.Label(self._inner_frame, text="Load a video list to see preview", anchor="center")
        self.preview_label.pack(fill="both", expand=True)

        self.preview_canvas.bind("<Configure>", lambda e: self._update_layout())

        # --- Zoom bindings ---
        for w in [self.preview_canvas, self.preview_label]:
            w.bind("<MouseWheel>", self._handle_zoom)
            w.bind("<Button-4>", self._handle_zoom)
            w.bind("<Button-5>", self._handle_zoom)

        # --- Drag/pan bindings ---
        self.preview_canvas.bind("<Button-1>", lambda e: self.preview_canvas.focus_set(), add="+")
        for b in ["<ButtonPress-1>", "<ButtonPress-2>"]:
            self.preview_label.bind(b, self._start_drag_scroll)
        for b in ["<B1-Motion>", "<B2-Motion>"]:
            self.preview_label.bind(b, self._drag_scroll)
        for b in ["<ButtonRelease-1>", "<ButtonRelease-2>"]:
            self.preview_label.bind(b, self._end_drag_scroll)

        # --- Key bindings ---
        # Bind on the root window for reliable key capture regardless of focus.
        # The canvas child may hold focus, preventing Toplevel-level bindings from firing.
        root = self.parent.winfo_toplevel()
        self._bind_f11 = root.bind("<F11>", lambda e: (self.toggle_fullscreen(), "break")[1], add="+")
        self._bind_escape = root.bind("<Escape>", lambda e: (self._on_escape(), "break")[1], add="+")
        self._bind_x_lower = root.bind("<x>", lambda e: (self._toggle_cross_eye(), "break")[1], add="+")
        self._bind_x_upper = root.bind("<X>", lambda e: (self._toggle_cross_eye(), "break")[1], add="+")

        # Unified key relay — pass keys to parent's top-level window
        self._relaying_lock = False
        self.bind("<KeyPress>", self._relay_key_event)

        # Double-click toggles fullscreen
        self.preview_label.bind("<Double-Button-1>", lambda e: self.toggle_fullscreen())

        # Close handler — hide instead of destroy
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Suppress initial geometry manager adjustments
        self.withdraw()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show(self):
        """Show (or re-show) the canvas window."""
        self.deiconify()
        self.lift()
        self.focus_set()

    def hide(self):
        """Hide the canvas window without destroying it."""
        self.withdraw()

    def destroy_window(self):
        """Fully destroy the Toplevel. After this the window cannot be re-shown."""
        # Remove root-level key bindings (targeted, using stored callback IDs)
        try:
            root = self.parent.winfo_toplevel()
            for seq, attr in (
                ("<F11>", "_bind_f11"),
                ("<Escape>", "_bind_escape"),
                ("<x>", "_bind_x_lower"),
                ("<X>", "_bind_x_upper"),
            ):
                cb_id = getattr(self, attr, None)
                if cb_id:
                    try:
                        root.unbind(seq, cb_id)
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            self.destroy()
        except Exception:
            pass

    def exists(self) -> bool:
        return self.winfo_exists()

    def update_frame(self, pil_image: Image.Image):
        """Display a PIL image on the canvas, centered if smaller than viewport."""
        if not self.winfo_exists():
            return
        try:
            self._pil_image = pil_image
            self._photo = ImageTk.PhotoImage(pil_image)
            self.preview_label.config(image=self._photo, text="")
            self.preview_label.image = self._photo
            self._update_layout()
        except Exception as e:
            logger.debug(f"PreviewCanvasWindow.update_frame error: {e}")

    def clear(self):
        """Clear the displayed image and show placeholder text."""
        dummy = ImageTk.PhotoImage(Image.new("RGBA", (1, 1), (0, 0, 0, 0)))
        self._photo = dummy
        self._pil_image = None
        try:
            self.preview_label.config(image=dummy, text="Load a video list to see preview")
            self.preview_label.image = dummy
        except Exception:
            pass

    def apply_theme(self, colors: dict):
        """Apply theme colors to the canvas."""
        try:
            bg = colors.get("bg", "#000000")
            self.preview_canvas.config(bg=bg, highlightthickness=0)
            self.configure(bg=bg)
        except Exception:
            pass

    def toggle_fullscreen(self, force: Optional[bool] = None):
        if force is None:
            self._is_fullscreen = not self._is_fullscreen
        else:
            self._is_fullscreen = bool(force)
        self.attributes("-fullscreen", self._is_fullscreen)

    @property
    def is_cross_eye(self) -> bool:
        if self.sbs_cross_eye_var is not None:
            return bool(self.sbs_cross_eye_var.get())
        return False

    # ------------------------------------------------------------------
    # Internal: layout
    # ------------------------------------------------------------------

    def _update_layout(self):
        """Center the image if smaller than canvas; show/hide scrollbars."""
        if self._pil_image is None or self._photo is None:
            self.v_scrollbar.grid_remove()
            self.h_scrollbar.grid_remove()
            return

        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()
        img_w = self._photo.width()
        img_h = self._photo.height()

        if canvas_w <= 1 or canvas_h <= 1:
            return

        v_needed = img_h > canvas_h
        h_needed = img_w > canvas_w

        if v_needed:
            self.v_scrollbar.grid()
        else:
            self.v_scrollbar.grid_remove()
        if h_needed:
            self.h_scrollbar.grid()
        else:
            self.h_scrollbar.grid_remove()

        x = max(0, (canvas_w - img_w) // 2)
        y = max(0, (canvas_h - img_h) // 2)

        self.preview_canvas.coords(self._canvas_window_id, x, y)
        self._inner_frame.update_idletasks()
        self.preview_canvas.config(scrollregion=(0, 0, max(canvas_w, img_w), max(canvas_h, img_h)))

    # ------------------------------------------------------------------
    # Internal: scroll
    # ------------------------------------------------------------------

    def _on_vscroll(self, *args):
        self.preview_canvas.yview(*args)

    def _on_hscroll(self, *args):
        self.preview_canvas.xview(*args)

    # ------------------------------------------------------------------
    # Internal: zoom
    # ------------------------------------------------------------------

    def _handle_zoom(self, event):
        """Delegate zoom to the zoom callback (typically on VideoPreviewer)."""
        if self._zoom_callback:
            self._zoom_callback(event)

    # ------------------------------------------------------------------
    # Internal: drag / pan
    # ------------------------------------------------------------------

    def _start_drag_scroll(self, event):
        if self.v_scrollbar.winfo_ismapped() or self.h_scrollbar.winfo_ismapped():
            self._is_dragging = True
            self.preview_canvas.config(cursor="fleur")
            self.preview_canvas.scan_mark(int(event.x), int(event.y))

    def _end_drag_scroll(self, event):
        self._is_dragging = False
        self.preview_canvas.config(cursor="")

    def _drag_scroll(self, event):
        if self._is_dragging:
            self.preview_canvas.scan_dragto(int(event.x), int(event.y), gain=1)

    # ------------------------------------------------------------------
    # Internal: cross-eye
    # ------------------------------------------------------------------

    def _toggle_cross_eye(self):
        if self.sbs_cross_eye_var is not None:
            self.sbs_cross_eye_var.set(not bool(self.sbs_cross_eye_var.get()))

        # Clear the display buffer so the next preview re-renders with new L/R order
        if hasattr(self.parent, "previewer") and hasattr(self.parent.previewer, "clear_display_buffer"):
            self.parent.previewer.clear_display_buffer()

        # Request preview refresh via parent if available
        if hasattr(self.parent, "on_slider_release"):
            self.parent.on_slider_release(None)
        elif hasattr(self.parent, "update_preview"):
            self.parent.update_preview()

    def _on_escape(self):
        """Handle Escape key: exit fullscreen if active."""
        if self._is_fullscreen:
            self.toggle_fullscreen(force=False)

    # ------------------------------------------------------------------
    # Internal: key relay
    # ------------------------------------------------------------------

    def _relay_key_event(self, event):
        """Pass key presses to the parent's top-level window."""
        if self._relaying_lock:
            return

        # X, F11, Escape are handled by root-window bindings (see _create_widgets).
        # Relay everything else to the parent's top-level window.
        root = self.parent.winfo_toplevel()
        kwargs = {
            "serial": event.serial,
            "time": event.time,
            "x": event.x,
            "y": event.y,
            "rootx": event.x_root,
            "rooty": event.y_root,
            "state": event.state,
            "keycode": event.keycode,
            "keysym": event.keysym,
        }
        try:
            self._relaying_lock = True
            if len(event.keysym) <= 1 or event.keysym in ("Left", "Right", "Up", "Down", "space", "Prior", "Next"):
                root.event_generate(f"<{event.keysym}>", **kwargs)
            else:
                root.event_generate("<KeyPress>", **kwargs)
        except Exception:
            pass
        finally:
            self._relaying_lock = False

        return "break"

    # ------------------------------------------------------------------
    # Internal: close
    # ------------------------------------------------------------------

    def _on_close(self):
        if self._on_close_callback:
            self._on_close_callback()
        self.hide()
