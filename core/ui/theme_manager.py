"""Theme management for StereoCrafter GUI applications.

Provides dark/light theme support with color palettes and styling utilities
for tkinter-based applications.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Dark theme color palette
DARK_COLORS = {
    "bg": "#2b2b2b",
    "fg": "white",
    "entry_bg": "#3c3c3c",
    "menu_bg": "#3c3c3c",
    "menu_fg": "white",
    "active_bg": "#555555",
    "active_fg": "white",
    "tooltip_bg": "#4a4a4a",
    "tooltip_fg": "white",
    "theme": "black",
    "panel_bg": "#3a4047",
    "panel_trough": "#262b31",
    "panel_bar": "#5d96e0",
}

# Light theme color palette
LIGHT_COLORS = {
    "bg": "#d9d9d9",
    "fg": "black",
    "entry_bg": "#ffffff",
    "menu_bg": "#f0f0f0",
    "menu_fg": "black",
    "active_bg": "#dddddd",
    "active_fg": "black",
    "tooltip_bg": "#ffffe0",
    "tooltip_fg": "black",
    "theme": "clam",
    "panel_bg": "#e6ebf2",
    "panel_trough": "#bcc6d3",
    "panel_bar": "#4f83cc",
}


def get_theme_colors(is_dark: bool) -> Dict[str, str]:
    """Get the color palette for the specified theme.

    Args:
        is_dark: True for dark theme, False for light theme

    Returns:
        Dictionary containing color values for the theme
    """
    return DARK_COLORS if is_dark else LIGHT_COLORS


def is_dark_mode(config: Dict[str, Any]) -> bool:
    """Check if dark mode is enabled in configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if dark mode is enabled
    """
    return bool(config.get("dark_mode_enabled", False))


class ThemeManager:
    """Manages application theming for dark/light mode switching.

    Provides utilities for applying themes to tkinter widgets and
    maintaining consistent styling across the application.

    Args:
        dark_mode_var: Optional tkinter BooleanVar tracking theme state
        config: Optional configuration dictionary
    """

    def __init__(self, dark_mode_var: Optional["tk.BooleanVar"] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the theme manager.

        Args:
            dark_mode_var: Tkinter BooleanVar tracking dark mode state
            config: Configuration dictionary
        """
        self.dark_mode_var = dark_mode_var
        self.config = config or {}

    def is_dark_mode(self) -> bool:
        """Check if dark mode is currently enabled.

        Returns:
            True if dark mode is enabled
        """
        if self.dark_mode_var:
            return bool(self.dark_mode_var.get())
        return is_dark_mode(self.config)

    def get_colors(self) -> Dict[str, str]:
        """Get the current theme colors.

        Returns:
            Dictionary containing current color palette
        """
        return get_theme_colors(self.is_dark_mode())

    def apply_theme_to_style(self, style: "ttk.Style", root_window: Optional["tk.Tk"] = None) -> None:
        """Apply the current theme to ttk styles.

        Args:
            style: ttk.Style object to configure
            root_window: Optional root window for bg configuration
        """
        colors = self.get_colors()

        # Apply theme to root window if provided
        if root_window:
            root_window.configure(bg=colors["bg"])
            # Use set_theme if it's a ThemedTk instance, otherwise theme_use
            if hasattr(root_window, "set_theme"):
                try:
                    root_window.set_theme(colors["theme"])
                except Exception as e:
                    logger.warning(f"Failed to set theme via set_theme: {e}")
                    style.theme_use(colors["theme"])
            else:
                try:
                    style.theme_use(colors["theme"])
                except Exception as e:
                    logger.warning(f"Failed to set theme via theme_use: {e}")

            # Crucial: let the theme switch take effect before configuring styles
            root_window.update_idletasks()

        # Configure basic styles for the current theme
        # We apply to common prefixes to be thorough
        bg = colors["bg"]
        fg = colors["fg"]
        entry_bg = colors["entry_bg"]

        for style_name in ["TFrame", "TLabelframe", "TLabel", "TCheckbutton", "TRadiobutton"]:
            style.configure(style_name, background=bg, foreground=fg)

        style.configure("TLabelframe.Label", background=bg, foreground=fg)

        # Configure style maps for interactive widgets
        style.map(
            "TCheckbutton",
            foreground=[("active", fg), ("!disabled", fg)],
            background=[("active", bg), ("!disabled", bg)],
        )
        style.map(
            "TRadiobutton",
            foreground=[("active", fg), ("!disabled", fg)],
            background=[("active", bg), ("!disabled", bg)],
        )

        # Configure Entry and Combobox
        # Some themes require explicit state mappings to override native looks
        style.map(
            "TEntry",
            fieldbackground=[("!disabled", entry_bg), ("focus", entry_bg)],
            foreground=[("!disabled", fg), ("focus", fg)],
        )
        style.configure("TEntry", insertcolor=fg)

        style.map(
            "TCombobox",
            fieldbackground=[("readonly", entry_bg), ("!disabled", entry_bg), ("focus", entry_bg)],
            foreground=[("readonly", fg), ("!disabled", fg), ("focus", fg)],
            selectbackground=[("readonly", entry_bg)],
            selectforeground=[("readonly", fg)],
        )

        # Force a final update to catch any delayed renders
        if root_window:
            root_window.update_idletasks()

    def apply_theme_to_menus(self, menus: list, menubar: Optional["tk.Menu"] = None) -> None:
        """Apply theme colors to menu widgets.

        Args:
            menus: List of tk.Menu widgets to configure
            menubar: Optional menubar widget
        """
        colors = self.get_colors()

        all_menus = list(menus)
        if menubar:
            all_menus.append(menubar)

        for menu in all_menus:
            try:
                menu.config(
                    bg=colors["menu_bg"],
                    fg=colors["menu_fg"],
                    activebackground=colors["active_bg"],
                    activeforeground=colors["active_fg"],
                )
            except Exception:
                pass

    def apply_theme_to_labels(self, labels: list, bg: Optional[str] = None, fg: Optional[str] = None) -> None:
        """Apply theme colors to label widgets.

        Args:
            labels: List of tk.Label widgets to configure
            bg: Optional background color override
            fg: Optional foreground color override
        """
        colors = self.get_colors()

        for label in labels:
            try:
                if bg:
                    label.config(bg=bg)
                else:
                    label.config(bg=colors["bg"])
                if fg:
                    label.config(fg=fg)
                else:
                    label.config(fg=colors["fg"])
            except Exception:
                pass

    def apply_theme_to_canvas(self, canvas: "tk.Canvas") -> None:
        """Apply theme background color to a canvas widget.

        Args:
            canvas: tk.Canvas widget to configure
        """
        colors = self.get_colors()
        try:
            canvas.config(bg=colors["bg"], highlightthickness=0)
        except Exception:
            pass

    def get_style_config(self) -> Dict[str, str]:
        """Get the current style configuration for persistence.

        Returns:
            Dictionary containing style-relevant configuration
        """
        colors = self.get_colors()
        return {
            "dark_mode_enabled": self.is_dark_mode(),
            "theme_bg": colors["bg"],
            "theme_fg": colors["fg"],
            "theme_entry_bg": colors["entry_bg"],
        }

    @staticmethod
    def get_available_themes() -> list:
        """Get list of available theme names.

        Returns:
            List of theme names
        """
        return ["default", "black", "clam", "alt", "classic"]

    def configure_compact_styles(self, style: "ttk.Style") -> None:
        """Configure compact button and entry styles for space-efficient UIs.

        Args:
            style: ttk.Style object to configure
        """
        colors = self.get_colors()
        panel_bg = colors.get("panel_bg", colors["bg"])

        try:
            style.configure("SmallTool.TButton", padding=(0, 0), anchor="center")
            style.configure("CompactAction.TButton", padding=(4, 0), anchor="center")
            style.configure("Loading.TButton", padding=(4, 0), anchor="center")
            style.configure("TEntry", padding=(1, 0))
            style.configure("TCombobox", padding=(1, 0), arrowsize=10)
            style.configure("TSpinbox", padding=(1, 0), arrowsize=10)
        except Exception:
            pass

    def configure_progressbar_style(
        self, style: "ttk.Style", trough_color: Optional[str] = None, bar_color: Optional[str] = None
    ) -> None:
        """Configure progress bar styles with custom colors.

        Args:
            style: ttk.Style object to configure
            trough_color: Optional color for the progress bar trough
            bar_color: Optional color for the progress bar fill
        """
        colors = self.get_colors()
        trough = trough_color or colors.get("panel_trough", colors["bg"])
        bar = bar_color or colors.get("panel_bar", "#5d96e0")

        try:
            style.configure(
                "Custom.Horizontal.TProgressbar",
                troughcolor=trough,
                background=bar,
                lightcolor=bar,
                darkcolor=bar,
                bordercolor=trough,
                thickness=12,
            )
            style.configure(
                "ProgressPanel.TLabel", background=colors.get("panel_bg", colors["bg"]), foreground=colors["fg"]
            )
        except Exception:
            pass

    def apply_theme_to_labelframes(self, labelframes: list) -> None:
        """Apply theme colors to a list of labelframe widgets.

        Args:
            labelframes: List of tk.LabelFrame widgets to configure
        """
        colors = self.get_colors()

        for lf in labelframes:
            if lf is not None:
                try:
                    lf.configure(bg=colors["bg"], fg=colors["fg"])
                except Exception:
                    pass

    def apply_theme_to_widgets_by_names(
        self, widget_dict: Dict[str, Any], bg: Optional[str] = None, fg: Optional[str] = None
    ) -> None:
        """Apply theme colors to widgets referenced by name dictionary.

        Args:
            widget_dict: Dictionary mapping widget names to widget objects
            bg: Optional background color override
            fg: Optional foreground color override
        """
        colors = self.get_colors()
        panel_bg = colors.get("panel_bg", colors["bg"])

        for w_name, w in widget_dict.items():
            if w is not None:
                try:
                    if fg and hasattr(w, "configure"):
                        w.configure(bg=bg or panel_bg, fg=fg)
                    else:
                        w.configure(bg=bg or panel_bg)
                except Exception:
                    try:
                        w.configure(bg=bg or panel_bg)
                    except Exception:
                        pass
