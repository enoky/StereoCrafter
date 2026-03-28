"""Reusable tkinter widgets and UI helper functions for StereoCrafter."""

import tkinter as tk
from tkinter import Toplevel, Label, ttk
from typing import Optional, Tuple, Callable

DEFAULT_TICK_RELY = 0.6
DEFAULT_TICK_RELHEIGHT = 0.6
DEFAULT_TICK_WIDTH = 2
DEFAULT_TICK_COLOR = "#6b7280"
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
        _update_marker_position()

    slider = ttk.Scale(
        parent, from_=0, to=total_steps, variable=internal_int_var, orient="horizontal", command=on_slider_move
    )
    slider.grid(row=row, column=1, sticky="ew", padx=2)

    value_label = ttk.Label(parent, text="", width=VALUE_LABEL_FIXED_WIDTH)
    value_label.grid(row=row, column=2, sticky="w", padx=0)
    parent.grid_columnconfigure(1, weight=1)

    if hasattr(GUI_self, "on_slider_release"):
        slider.bind("<ButtonRelease-1>", GUI_self.on_slider_release)

    default_marker_label = None

    def _draw_default_marker():
        nonlocal default_marker_label
        if default_value is None:
            return
        if default_marker_label is not None:
            try:
                default_marker_label.destroy()
            except Exception:
                pass
            default_marker_label = None
        try:
            slider.update_idletasks()
            slider_width = slider.winfo_width()
            if slider_width <= 1:
                return
            range_ratio = (default_value - from_) / (to - from_)
            range_ratio = max(0.0, min(1.0, range_ratio))
            knob_offset = 15
            track_start = knob_offset
            track_end = slider_width - knob_offset
            track_width = track_end - track_start
            x_pos = track_start + int(track_width * range_ratio)
            marker_canvas = tk.Canvas(slider, width=4, height=10, highlightthickness=0)
            marker_canvas.create_line(2, 0, 2, 10, fill=DEFAULT_TICK_COLOR, width=2)
            marker_canvas.place(x=x_pos, y=slider.winfo_height() - 2, anchor="s")
            default_marker_label = marker_canvas
        except Exception as e:
            print(f"Marker error: {e}")

    def _update_marker_position():
        if default_marker_label is None or default_value is None:
            return
        try:
            slider_width = slider.winfo_width()
            if slider_width <= 1:
                return
            range_ratio = (default_value - from_) / (to - from_)
            range_ratio = max(0.0, min(1.0, range_ratio))
            knob_offset = 15
            track_start = knob_offset
            track_end = slider_width - knob_offset
            track_width = track_end - track_start
            x_pos = track_start + int(track_width * range_ratio)
            default_marker_label.place_configure(x=x_pos, y=slider.winfo_height() - 2, anchor="s")
        except Exception as e:
            print(f"Update marker error: {e}")

    def _on_right_click(event):
        if default_value is not None:
            var.set(default_value)
            sync_external_change()
            _update_marker_position()
            if hasattr(GUI_self, "on_slider_release"):
                GUI_self.on_slider_release(event)
        return "break"

    def _on_middle_click(event):
        try:
            slider_width = slider.winfo_width()
            if slider_width <= 1:
                return "break"
            click_x = event.x
            rel_x = click_x / slider_width
            rel_x = max(0.0, min(1.0, rel_x))
            clicked_notch = int(rel_x * total_steps)
            clicked_notch = max(0, min(total_steps, clicked_notch))
            clicked_val = from_ + (clicked_notch * actual_step)
            clicked_val = max(from_, min(to, clicked_val))
            var.set(clicked_val)
            sync_external_change()
            if hasattr(GUI_self, "on_slider_release"):
                GUI_self.on_slider_release(event)
        except Exception:
            pass
        return "break"

    if default_value is not None:
        slider.bind("<Button-3>", _on_right_click)
        slider.bind("<ButtonRelease-3>", _on_right_click)
        slider.bind("<Button-2>", _on_middle_click)
        slider.bind("<ButtonRelease-2>", _on_middle_click)
        GUI_self.after(50, _draw_default_marker)

    def _on_configure(event):
        if default_marker_label is None:
            _draw_default_marker()

    slider.bind("<Configure>", _on_configure)

    def sync_external_change():
        try:
            current_f = float(var.get())
            new_notch = int((current_f - from_) / actual_step)
            internal_int_var.set(new_notch)
            update_label_only(current_f)
            _update_marker_position()
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
        GUI_self,
        x_frame,
        text_x,
        var_x,
        f_x,
        t_x,
        0,
        decimals=d_x,
        tooltip_key=tooltip_key_x,
        step_size=step_size_x,
        default_value=default_x,
    )

    y_frame = ttk.Frame(xy_frame)
    y_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
    y_frame.grid_columnconfigure(1, weight=1)
    set_y = create_single_slider_with_label_updater(
        GUI_self,
        y_frame,
        text_y,
        var_y,
        f_y,
        t_y,
        0,
        decimals=d_y,
        tooltip_key=tooltip_key_y,
        step_size=step_size_y,
        default_value=default_y,
    )
    return xy_frame, (set_x, set_y), (x_frame, y_frame)
