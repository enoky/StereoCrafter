"""Reusable tkinter widgets and UI helper functions for StereoCrafter."""

import tkinter as tk
from tkinter import Toplevel, Label, ttk
from typing import Optional, Tuple, Callable

# --- Slider default tick marker appearance ---
DEFAULT_TICK_RELY = 0.6
DEFAULT_TICK_RELHEIGHT = 0.6
DEFAULT_TICK_WIDTH = 2
DEFAULT_TICK_COLOR = "#6b7280"
DEFAULT_TICK_TRACK_PAD_PCT = 0.0
DEFAULT_TICK_X_OFFSET_PX = 5

class Tooltip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.show_delay = 600
        self.hide_delay = 100
        self.enter_id = None
        self.leave_id = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
        self.widget.bind("<ButtonPress>", self.hide_tooltip)

    def _display_tooltip(self):
        if self.tooltip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 20
        self.tooltip_window = Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = Label(
            self.tooltip_window,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            justify="left",
            wraplength=250,
        )
        label.pack(ipadx=1)

    def hide_tooltip(self, event=None):
        if self.enter_id:
            self.widget.after_cancel(self.enter_id)
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

    def show_tooltip(self, event=None):
        if self.leave_id:
            self.widget.after_cancel(self.leave_id)
        self.enter_id = self.widget.after(self.show_delay, self._display_tooltip)

def create_single_slider_with_label_updater(
    GUI_self,
    parent: ttk.Frame,
    text: str,
    var: tk.Variable,
    from_: float,
    to: float,
    row: int,
    decimals: int = 0,
    tooltip_key: Optional[str] = None,
    trough_increment: float = -1.0,
    display_next_odd_integer: bool = False,
    custom_label_formula: Optional[Callable] = None,
    step_size: Optional[float] = None,
    default_value: Optional[float] = None,
) -> None:
    """Creates a single slider using Discrete Step Mapping."""
    VALUE_LABEL_FIXED_WIDTH = 5

    label = ttk.Label(parent, text=text, anchor="e")
    label.grid(row=row, column=0, sticky="ew", padx=0, pady=2)

    if tooltip_key and hasattr(GUI_self, "_create_hover_tooltip"):
        GUI_self._create_hover_tooltip(label, tooltip_key)

    actual_step = step_size if step_size is not None else (0.5 if decimals > 0 else 1.0)
    total_steps = int((to - from_) / actual_step)
    internal_int_var = tk.IntVar(value=int((float(var.get()) - from_) / actual_step))

    def update_label_only(value_float: float) -> None:
        try:
            if custom_label_formula:
                value_label.config(text=custom_label_formula(value_float))
                return

            display_value = value_float
            if display_next_odd_integer:
                k_int = int(round(value_float))
                if k_int > 0 and k_int % 2 == 0:
                    display_value = k_int + 1
                elif k_int > 0:
                    display_value = k_int
                elif k_int == 0:
                    display_value = 0

            value_label.config(text=f"{display_value:.{decimals}f}")
        except Exception:
            pass

    def on_slider_move(val):
        notch = int(float(val))
        actual_val = from_ + (notch * actual_step)
        actual_val = max(from_, min(to, actual_val))
        var.set(actual_val)
        update_label_only(actual_val)

    slider = ttk.Scale(
        parent, from_=0, to=total_steps, variable=internal_int_var, orient="horizontal", command=on_slider_move
    )
    slider.grid(row=row, column=1, sticky="ew", padx=2)

    value_label = ttk.Label(parent, text="", width=VALUE_LABEL_FIXED_WIDTH)
    value_label.grid(row=row, column=2, sticky="w", padx=0)
    parent.grid_columnconfigure(1, weight=1)

    if hasattr(GUI_self, "on_slider_release"):
        slider.bind("<ButtonRelease-1>", GUI_self.on_slider_release)

    def sync_external_change():
        try:
            current_f = float(var.get())
            new_notch = int((current_f - from_) / actual_step)
            internal_int_var.set(new_notch)
            update_label_only(current_f)
        except Exception:
            pass

    sync_external_change()

    if hasattr(GUI_self, "slider_label_updaters"):
        GUI_self.slider_label_updaters.append(sync_external_change)
    if hasattr(GUI_self, "widgets_to_disable"):
        GUI_self.widgets_to_disable.append(slider)

    return lambda val: (var.set(val), sync_external_change())

def create_dual_slider_layout(
    GUI_self,
    parent: ttk.Frame,
    text_x: str,
    text_y: str,
    var_x: tk.Variable,
    var_y: tk.Variable,
    from_: float,
    to: float,
    row: int,
    decimals: int = 0,
    is_integer: bool = True,
    tooltip_key_x: Optional[str] = None,
    tooltip_key_y: Optional[str] = None,
    trough_increment: float = -1,
    display_next_odd_integer: bool = False,
    custom_label_formula: Optional[Callable] = None,
    default_x: Optional[float] = None,
    default_y: Optional[float] = None,
    from_y: Optional[float] = None,
    to_y: Optional[float] = None,
    decimals_y: Optional[int] = None,
    step_size_x: Optional[float] = None,
    step_size_y: Optional[float] = None,
) -> Tuple[ttk.Frame, Tuple[Callable, Callable], Tuple[ttk.Frame, ttk.Frame]]:
    """Creates a two-column (X/Y) slider row."""
    xy_frame = ttk.Frame(parent)
    xy_frame.grid(row=row, column=0, columnspan=2, sticky="ew", padx=5, pady=0)
    xy_frame.grid_columnconfigure(0, weight=1)
    xy_frame.grid_columnconfigure(1, weight=1)

    f_x, t_x, d_x = from_, to, decimals
    f_y = from_y if from_y is not None else from_
    t_y = to_y if to_y is not None else to
    d_y = decimals_y if decimals_y is not None else decimals

    x_frame = ttk.Frame(xy_frame)
    x_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
    x_frame.grid_columnconfigure(1, weight=1)
    set_x = create_single_slider_with_label_updater(
        GUI_self, x_frame, text_x, var_x, f_x, t_x, 0, decimals=d_x, tooltip_key=tooltip_key_x, step_size=step_size_x, default_value=default_x
    )

    y_frame = ttk.Frame(xy_frame)
    y_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    y_frame.grid_columnconfigure(1, weight=1)
    set_y = create_single_slider_with_label_updater(
        GUI_self, y_frame, text_y, var_y, f_y, t_y, 0, decimals=d_y, tooltip_key=tooltip_key_y, step_size=step_size_y, default_value=default_y
    )
    return xy_frame, (set_x, set_y), (x_frame, y_frame)
