"""GUI widget creation utilities for Splatting GUI.

Provides helper functions for creating common widget patterns used
in the Splatting GUI, including labeled entries, folder selection rows,
checkbox groups, and sliders with tooltips.
"""

import logging
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class Tooltip:
    """Tooltip helper class for tkinter widgets.

    Creates a simple tooltip that appears when hovering over a widget.

    Args:
        widget: The widget to attach the tooltip to
        text: The tooltip text content
    """

    def __init__(self, widget: tk.Widget, text: str):
        """Initialize the tooltip.

        Args:
            widget: The widget to attach the tooltip to
            text: The tooltip text content
        """
        self.widget = widget
        self.text = text
        self.tooltip_window: Optional[tk.Toplevel] = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        """Display the tooltip."""
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25

        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip_window,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe0",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Segoe UI", 9),
        )
        label.pack()

    def _hide(self, event=None):
        """Hide the tooltip."""
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


def create_tooltip(widget: tk.Widget, text: str) -> Tooltip:
    """Create and attach a tooltip to a widget.

    Args:
        widget: The widget to attach the tooltip to
        text: The tooltip text content

    Returns:
        Tooltip instance
    """
    return Tooltip(widget, text)


def create_labeled_entry(
    parent: tk.Widget,
    label_text: str,
    textvariable: tk.Variable,
    tooltip_text: Optional[str] = None,
    entry_width: int = 40,
    row: int = 0,
    column: int = 0,
    sticky: str = "ew",
    padx: int = 5,
    pady: int = 2,
) -> Tuple[ttk.Label, ttk.Entry]:
    """Create a label and entry pair.

    Creates a standard label and entry widget pair, commonly used for
    text input fields in the GUI.

    Args:
        parent: Parent widget
        label_text: Text for the label
        textvariable: Tkinter variable for the entry
        tooltip_text: Optional tooltip text for both widgets
        entry_width: Width of the entry widget
        row: Grid row for label
        column: Grid column for label (entry goes in column+1)
        sticky: Sticky option for grid
        padx: Horizontal padding
        pady: Vertical padding

    Returns:
        Tuple of (label_widget, entry_widget)
    """
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=column, sticky="e", padx=padx, pady=pady)

    entry = ttk.Entry(parent, textvariable=textvariable, width=entry_width)
    entry.grid(row=row, column=column + 1, sticky=sticky, padx=padx, pady=pady)

    if tooltip_text:
        create_tooltip(label, tooltip_text)
        create_tooltip(entry, tooltip_text)

    return label, entry


def create_folder_selection_row(
    parent: tk.Widget,
    label_text: str,
    textvariable: tk.StringVar,
    browse_folder_callback: Callable,
    select_file_callback: Optional[Callable] = None,
    file_types: Optional[List[Tuple[str, str]]] = None,
    tooltip_key: Optional[str] = None,
    row: int = 0,
    create_tooltips_func: Optional[Callable] = None,
) -> Dict[str, tk.Widget]:
    """Create a folder selection row with label, entry, and browse buttons.

    Creates the standard pattern of: Label | Entry | Browse Folder | [Select File]
    Used for input/output folder selection throughout the GUI.

    Args:
        parent: Parent widget (usually a LabelFrame)
        label_text: Text for the label
        textvariable: StringVar for the entry
        browse_folder_callback: Callback for folder browse button
        select_file_callback: Optional callback for file select button
        file_types: Optional file types for file dialog [(description, pattern), ...]
        tooltip_key: Key for tooltip lookup (if create_tooltips_func provided)
        row: Grid row
        create_tooltips_func: Optional function to create tooltips by key

    Returns:
        Dictionary of created widgets
    """
    widgets = {}

    # Label
    lbl = ttk.Label(parent, text=label_text)
    lbl.grid(row=row, column=0, sticky="e", padx=5, pady=0)
    widgets["label"] = lbl

    # Entry
    entry = ttk.Entry(parent, textvariable=textvariable)
    entry.grid(row=row, column=1, padx=5, pady=0, sticky="ew")
    widgets["entry"] = entry

    # Browse Folder button
    btn_folder = ttk.Button(parent, text="Browse Folder", command=browse_folder_callback)
    btn_folder.grid(row=row, column=2, padx=2, pady=0)
    widgets["btn_browse_folder"] = btn_folder

    # Select File button (optional)
    if select_file_callback:
        btn_file = ttk.Button(parent, text="Select File", command=select_file_callback)
        btn_file.grid(row=row, column=3, padx=2, pady=0)
        widgets["btn_select_file"] = btn_file

        if create_tooltips_func and tooltip_key:
            create_tooltips_func(lbl, tooltip_key)
            create_tooltips_func(entry, tooltip_key)
            create_tooltips_func(btn_folder, f"{tooltip_key}_folder")
            create_tooltips_func(btn_file, f"{tooltip_key}_file")
    else:
        if create_tooltips_func and tooltip_key:
            create_tooltips_func(lbl, tooltip_key)
            create_tooltips_func(entry, tooltip_key)
            create_tooltips_func(btn_folder, tooltip_key)

    return widgets


def create_checkbox_group(
    parent: tk.Widget,
    checkboxes: List[Dict[str, Any]],
    row: int = 0,
    columns: int = 3,
) -> List[ttk.Checkbutton]:
    """Create a group of checkboxes in a grid layout.

    Args:
        parent: Parent widget
        checkboxes: List of checkbox configs with keys:
            - text: Checkbox label
            - variable: BooleanVar
            - command: Optional callback
            - tooltip: Optional tooltip text
            - width: Optional width
        row: Starting grid row
        columns: Number of columns in the grid

    Returns:
        List of created Checkbutton widgets
    """
    created = []

    for i, config in enumerate(checkboxes):
        col = i % columns
        current_row = row + (i // columns)

        cb = ttk.Checkbutton(
            parent,
            text=config.get("text", ""),
            variable=config.get("variable"),
            command=config.get("command"),
            width=config.get("width", None),
        )
        cb.grid(row=current_row, column=col, sticky="w", padx=5, pady=2)

        if config.get("tooltip"):
            create_tooltip(cb, config["tooltip"])

        created.append(cb)

    return created


def create_labeled_slider(
    parent: tk.Widget,
    label_text: str,
    variable: tk.Variable,
    from_value: float,
    to_value: float,
    resolution: float = 0.1,
    orient: str = tk.HORIZONTAL,
    length: int = 200,
    tooltip_text: Optional[str] = None,
    command: Optional[Callable] = None,
    show_value: bool = True,
    row: int = 0,
    column: int = 0,
) -> Tuple[ttk.Label, ttk.Scale, Optional[ttk.Label]]:
    """Create a labeled slider with optional value display.

    Args:
        parent: Parent widget
        label_text: Text for the label
        variable: Tkinter variable (DoubleVar or IntVar)
        from_value: Minimum value
        to_value: Maximum value
        resolution: Step size
        orient: Orientation (HORIZONTAL or VERTICAL)
        length: Length of the slider in pixels
        tooltip_text: Optional tooltip text
        command: Optional callback when value changes
        show_value: Whether to show value label next to slider
        row: Grid row
        column: Grid column

    Returns:
        Tuple of (label_widget, slider_widget, value_label or None)
    """
    # Label
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=column, sticky="e", padx=5, pady=2)

    # Slider frame (to hold slider + value)
    slider_frame = ttk.Frame(parent)
    slider_frame.grid(row=row, column=column + 1, sticky="ew", padx=5, pady=2)
    slider_frame.grid_columnconfigure(0, weight=1)

    # Slider
    slider = ttk.Scale(
        slider_frame,
        from_=from_value,
        to=to_value,
        orient=orient,
        length=length,
        variable=variable,
        command=command,
    )
    slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

    # Value label (optional)
    value_label = None
    if show_value:
        value_label = ttk.Label(slider_frame, textvariable=variable, width=6)
        value_label.pack(side=tk.LEFT, padx=(5, 0))

    if tooltip_text:
        create_tooltip(label, tooltip_text)
        create_tooltip(slider, tooltip_text)
        if value_label:
            create_tooltip(value_label, tooltip_text)

    return label, slider, value_label


def create_button_group(
    parent: tk.Widget,
    buttons: List[Dict[str, Any]],
    row: int = 0,
    column: int = 0,
    orientation: str = "horizontal",
) -> List[ttk.Button]:
    """Create a group of buttons.

    Args:
        parent: Parent widget
        buttons: List of button configs with keys:
            - text: Button label
            - command: Callback function
            - tooltip: Optional tooltip text
            - state: Optional initial state ("normal", "disabled")
            - width: Optional width
        row: Grid row
        column: Grid column
        orientation: "horizontal" or "vertical"

    Returns:
        List of created Button widgets
    """
    created = []
    frame = ttk.Frame(parent)
    frame.grid(row=row, column=column, sticky="ew", padx=5, pady=2)

    for i, config in enumerate(buttons):
        btn = ttk.Button(
            frame,
            text=config.get("text", ""),
            command=config.get("command"),
            width=config.get("width", None),
        )

        if config.get("state"):
            btn.config(state=config["state"])

        if orientation == "horizontal":
            btn.pack(side=tk.LEFT, padx=2)
        else:
            btn.pack(side=tk.TOP, pady=2)

        if config.get("tooltip"):
            create_tooltip(btn, config["tooltip"])

        created.append(btn)

    return created


def create_section_frame(
    parent: tk.Widget,
    title: str,
    row: int = 0,
    column: int = 0,
    columnspan: int = 1,
    padx: int = 10,
    pady: int = 2,
) -> ttk.LabelFrame:
    """Create a labeled frame section.

    Args:
        parent: Parent widget
        title: Frame title text
        row: Grid row
        column: Grid column
        columnspan: Number of columns to span
        padx: Horizontal padding
        pady: Vertical padding

    Returns:
        Created LabelFrame widget
    """
    frame = ttk.LabelFrame(parent, text=title)
    frame.grid(
        row=row, column=column, columnspan=columnspan, sticky="nsew", padx=padx, pady=pady
    )
    return frame


def create_dropdown(
    parent: tk.Widget,
    label_text: str,
    variable: tk.StringVar,
    values: List[str],
    command: Optional[Callable] = None,
    tooltip_text: Optional[str] = None,
    row: int = 0,
    column: int = 0,
    width: int = 20,
) -> Tuple[ttk.Label, ttk.Combobox]:
    """Create a labeled dropdown (Combobox).

    Args:
        parent: Parent widget
        label_text: Text for the label
        variable: StringVar for the dropdown
        values: List of dropdown values
        command: Optional callback when selection changes
        tooltip_text: Optional tooltip text
        row: Grid row
        column: Grid column
        width: Width of the combobox

    Returns:
        Tuple of (label_widget, combobox_widget)
    """
    label = ttk.Label(parent, text=label_text)
    label.grid(row=row, column=column, sticky="e", padx=5, pady=2)

    combobox = ttk.Combobox(
        parent,
        textvariable=variable,
        values=values,
        width=width,
        state="readonly",
    )
    combobox.grid(row=row, column=column + 1, sticky="w", padx=5, pady=2)

    if command:
        combobox.bind("<<ComboboxSelected>>", lambda e: command())

    if tooltip_text:
        create_tooltip(label, tooltip_text)
        create_tooltip(combobox, tooltip_text)

    return label, combobox


def configure_grid_weights(
    widget: tk.Widget,
    column_weights: Optional[List[int]] = None,
    row_weights: Optional[List[int]] = None,
) -> None:
    """Configure grid column and row weights for responsive layout.

    Args:
        widget: The widget whose grid to configure
        column_weights: List of weights for each column (None = 1 for all)
        row_weights: List of weights for each row (None = no configuration)
    """
    if column_weights:
        for i, weight in enumerate(column_weights):
            widget.grid_columnconfigure(i, weight=weight)

    if row_weights:
        for i, weight in enumerate(row_weights):
            widget.grid_rowconfigure(i, weight=weight)


# Predefined tooltip texts for common widgets
COMMON_TOOLTIPS = {
    "input_source_clips": "Folder or file containing source video clips",
    "input_source_clips_folder": "Browse for a folder containing source video clips",
    "input_source_clips_file": "Select a single source video file",
    "input_depth_maps": "Folder or file containing depth maps (*.mp4, *.npz)",
    "input_depth_maps_folder": "Browse for a folder containing depth maps",
    "input_depth_maps_file": "Select a single depth map file",
    "output_splatted": "Output folder for splatted (right-eye) videos",
    "enable_full_res": "Enable full resolution output processing",
    "enable_low_res": "Enable low resolution output processing",
    "full_res_batch_size": "Number of frames to process per batch at full resolution",
    "low_res_batch_size": "Number of frames to process per batch at low resolution",
    "max_disp": "Maximum disparity (3D depth intensity). Typical range: 5-40.",
    "convergence": "Zero parallax plane position (0.0=foreground, 1.0=background)",
    "depth_gamma": "Gamma correction for depth map (1.0=linear, <1.0=compress far, >1.0=emphasize far)",
    "dual_output": "Output both left and right eye videos (for external 3D encoding)",
    "multi_map": "Enable multiple depth map selection for comparison",
}


def get_common_tooltip(key: str) -> Optional[str]:
    """Get predefined tooltip text for a common key.

    Args:
        key: Tooltip key

    Returns:
        Tooltip text or None if not found
    """
    return COMMON_TOOLTIPS.get(key)
