import tkinter as tk
from tkinter import ttk
from typing import Optional, Dict, Any, Callable
from core.common.encoding_utils import (
    QUALITY_PRESETS,
    CPU_TUNE_OPTIONS,
    ENCODER_OPTIONS,
    CODEC_OPTIONS,
    DEFAULT_ENCODING_CONFIG,
    build_encoder_args,
)

COLOR_TAGS_OPTIONS = ["Off", "Auto", "BT.709 L", "BT.709 F", "BT.2020 PQ", "BT.2020 HLG"]


class EncodingSettingsDialog:
    """Reusable encoding settings dialog for StereoCrafter GUIs.

    This dialog provides a comprehensive UI for configuring video encoding
    options including encoder, quality preset, tune, CRF, NVENC-specific
    options, and optionally DNxHR options and dual CRF fields.
    """

    def __init__(
        self,
        parent: tk.Tk,
        app_config: Optional[Dict[str, Any]] = None,
        help_data: Optional[Dict[str, str]] = None,
        title: str = "Encoding Settings",
        show_extra_options: bool = True,
        show_dual_crf: bool = False,
        show_color_tags: bool = True,
    ):
        """Initialize the encoding settings dialog.

        Args:
            parent: Parent tkinter window
            app_config: Optional existing config dict to populate values from
            help_data: Optional dict of help text strings
            title: Dialog window title
            show_extra_options: If True, show DNxHR options (for splatting).
            show_dual_crf: If True, show Full Res CRF and Low Res CRF fields (for splatting).
                          If False, show single CRF field.
            show_color_tags: If True, show Color Tags dropdown.
        """
        self.parent = parent
        self.app_config = app_config or {}
        self.help_data = help_data or {}
        self.title = title
        self.show_extra_options = show_extra_options
        self.show_dual_crf = show_dual_crf
        self.show_color_tags = show_color_tags

        self.result = None
        self._setup_variables()
        self._create_dialog()

    def _setup_variables(self):
        """Initialize tkinter variables from app config."""
        config = self.app_config

        self.codec_var = tk.StringVar(value=config.get("codec", "H.265"))
        self.encoder_var = tk.StringVar(value=config.get("encoding_encoder", "Auto"))
        self.quality_var = tk.StringVar(value=config.get("encoding_quality", DEFAULT_ENCODING_CONFIG["quality"]))
        self.tune_var = tk.StringVar(value=config.get("encoding_tune", DEFAULT_ENCODING_CONFIG["tune"]))

        if self.show_dual_crf:
            self.crf_full_var = tk.StringVar(
                value=str(config.get("output_crf_full", config.get("output_crf", DEFAULT_ENCODING_CONFIG["crf"])))
            )
            self.crf_low_var = tk.StringVar(
                value=str(config.get("output_crf_low", config.get("output_crf", DEFAULT_ENCODING_CONFIG["crf"])))
            )
        else:
            self.crf_var = tk.StringVar(
                value=str(config.get("output_crf", config.get("crf", DEFAULT_ENCODING_CONFIG["crf"])))
            )

        self.nvenc_lookahead_enabled_var = tk.BooleanVar(
            value=config.get("nvenc_lookahead_enabled", DEFAULT_ENCODING_CONFIG["nvenc_lookahead_enabled"])
        )
        self.nvenc_lookahead_var = tk.IntVar(
            value=config.get("nvenc_lookahead", DEFAULT_ENCODING_CONFIG["nvenc_lookahead"])
        )
        self.nvenc_spatial_aq_var = tk.BooleanVar(
            value=config.get("nvenc_spatial_aq", DEFAULT_ENCODING_CONFIG["nvenc_spatial_aq"])
        )
        self.nvenc_temporal_aq_var = tk.BooleanVar(
            value=config.get("nvenc_temporal_aq", DEFAULT_ENCODING_CONFIG["nvenc_temporal_aq"])
        )
        self.nvenc_aq_strength_var = tk.IntVar(
            value=config.get("nvenc_aq_strength", DEFAULT_ENCODING_CONFIG["nvenc_aq_strength"])
        )

        self.dnxhr_fullres_split_var = tk.BooleanVar(value=config.get("dnxhr_fullres_split", False))
        self.dnxhr_profile_var = tk.StringVar(value=config.get("dnxhr_profile", "HQX (10-bit 4:2:2)"))

        self.color_tags_var = tk.StringVar(value=config.get("color_tags", config.get("color_tags_mode", "Auto")))

    def _create_dialog(self):
        """Create the dialog window and widgets."""
        self.dialog = tk.Toplevel(self.parent)
        self.dialog.title(self.title)
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        self.dialog.resizable(False, False)

        outer = ttk.Frame(self.dialog, padding=15)
        outer.pack(fill="both", expand=True)

        row = 0

        lbl_codec = ttk.Label(outer, text="Codec:")
        lbl_codec.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
        self._add_tooltip(lbl_codec, "encoding_codec")

        cb_codec = ttk.Combobox(outer, textvariable=self.codec_var, values=CODEC_OPTIONS, state="readonly", width=18)
        cb_codec.grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        lbl_encoder = ttk.Label(outer, text="Encoder:")
        lbl_encoder.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
        self._add_tooltip(lbl_encoder, "encoding_encoder")

        cb_encoder = ttk.Combobox(
            outer, textvariable=self.encoder_var, values=ENCODER_OPTIONS, state="readonly", width=18
        )
        cb_encoder.grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        lbl_quality = ttk.Label(outer, text="Quality Preset:")
        lbl_quality.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
        self._add_tooltip(lbl_quality, "encoding_quality")

        cb_quality = ttk.Combobox(
            outer, textvariable=self.quality_var, values=QUALITY_PRESETS, state="readonly", width=18
        )
        cb_quality.grid(row=row, column=1, sticky="w", pady=4)

        row += 1
        lbl_tune = ttk.Label(outer, text="CPU Tune:")
        lbl_tune.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
        self._add_tooltip(lbl_tune, "encoding_tune")

        cb_tune = ttk.Combobox(outer, textvariable=self.tune_var, values=CPU_TUNE_OPTIONS, state="readonly", width=18)
        cb_tune.grid(row=row, column=1, sticky="w", pady=4)

        row += 1

        if self.show_dual_crf:
            lbl_crf_full = ttk.Label(outer, text="Full Res CRF:")
            lbl_crf_full.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
            self._add_tooltip(lbl_crf_full, "output_crf")

            entry_crf_full = ttk.Entry(outer, textvariable=self.crf_full_var, width=10)
            entry_crf_full.grid(row=row, column=1, sticky="w", pady=4)

            row += 1

            lbl_crf_low = ttk.Label(outer, text="Low Res CRF:")
            lbl_crf_low.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
            self._add_tooltip(lbl_crf_low, "output_crf")

            entry_crf_low = ttk.Entry(outer, textvariable=self.crf_low_var, width=10)
            entry_crf_low.grid(row=row, column=1, sticky="w", pady=4)
        else:
            lbl_crf = ttk.Label(outer, text="CRF (quality):")
            lbl_crf.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
            self._add_tooltip(lbl_crf, "output_crf")

            entry_crf = ttk.Entry(outer, textvariable=self.crf_var, width=10)
            entry_crf.grid(row=row, column=1, sticky="w", pady=4)

        row += 1

        if self.show_color_tags:
            lbl_color_tags = ttk.Label(outer, text="Color Tags:")
            lbl_color_tags.grid(row=row, column=0, sticky="e", padx=(0, 8), pady=4)
            self._add_tooltip(lbl_color_tags, "color_tags_mode")

            cb_color_tags = ttk.Combobox(
                outer, textvariable=self.color_tags_var, values=COLOR_TAGS_OPTIONS, state="readonly", width=18
            )
            cb_color_tags.grid(row=row, column=1, sticky="w", pady=4)
            row += 1

        self._create_nvenc_frame(outer, row)

        extra_row = row + 1

        if self.show_extra_options:
            self._create_dnxhr_frame(outer, extra_row)
            extra_row += 1

        self._create_buttons(outer, extra_row)

        self.dialog.update_idletasks()
        width = self.dialog.winfo_reqwidth()
        height = self.dialog.winfo_reqheight()

        parent_x = self.parent.winfo_rootx()
        parent_y = self.parent.winfo_rooty()

        # Position slightly to the right of the left edge and just under the menubar (which is usually represented by rooty)
        x = parent_x + 20
        y = parent_y + 10
        
        # Ensure the dialog is not positioned off-screen
        x = max(0, x)
        y = max(0, y)

        self.dialog.geometry(f"{width}x{height}+{x}+{y}")

    def _create_nvenc_frame(self, parent: ttk.Frame, row: int):
        """Create the NVENC options frame."""
        nv_frame = ttk.LabelFrame(parent, text="NVENC Options (only when NVENC is used)", padding=10)
        nv_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        nv_frame.grid_columnconfigure(1, weight=1)

        nv_row = 0

        chk_la = ttk.Checkbutton(nv_frame, variable=self.nvenc_lookahead_enabled_var)
        chk_la.grid(row=nv_row, column=0, sticky="w", pady=2)
        lbl_la = ttk.Label(nv_frame, text="Enable Lookahead")
        lbl_la.grid(row=nv_row, column=1, sticky="w", pady=2)
        self._add_tooltip(lbl_la, "encoding_nvenc_lookahead_enabled")

        nv_row += 1
        lbl_laf = ttk.Label(nv_frame, text="Lookahead Frames:")
        lbl_laf.grid(row=nv_row, column=0, sticky="e", padx=(0, 8), pady=2)
        self._add_tooltip(lbl_laf, "encoding_nvenc_lookahead")

        sp_la = ttk.Spinbox(nv_frame, from_=0, to=64, increment=1, textvariable=self.nvenc_lookahead_var, width=8)
        sp_la.grid(row=nv_row, column=1, sticky="w", pady=2)

        nv_row += 1
        chk_saq = ttk.Checkbutton(nv_frame, variable=self.nvenc_spatial_aq_var)
        chk_saq.grid(row=nv_row, column=0, sticky="w", pady=2)
        lbl_saq = ttk.Label(nv_frame, text="Spatial AQ")
        lbl_saq.grid(row=nv_row, column=1, sticky="w", pady=2)
        self._add_tooltip(lbl_saq, "encoding_nvenc_spatial_aq")

        nv_row += 1
        chk_taq = ttk.Checkbutton(nv_frame, variable=self.nvenc_temporal_aq_var)
        chk_taq.grid(row=nv_row, column=0, sticky="w", pady=2)
        lbl_taq = ttk.Label(nv_frame, text="Temporal AQ")
        lbl_taq.grid(row=nv_row, column=1, sticky="w", pady=2)
        self._add_tooltip(lbl_taq, "encoding_nvenc_temporal_aq")

        nv_row += 1
        lbl_aq = ttk.Label(nv_frame, text="AQ Strength:")
        lbl_aq.grid(row=nv_row, column=0, sticky="e", padx=(0, 8), pady=2)
        self._add_tooltip(lbl_aq, "encoding_nvenc_aq_strength")

        sp_aq = ttk.Spinbox(nv_frame, from_=1, to=15, increment=1, textvariable=self.nvenc_aq_strength_var, width=8)
        sp_aq.grid(row=nv_row, column=1, sticky="w", pady=2)

        self.nvenc_frame = nv_frame
        self.nvenc_lookahead_spinbox = sp_la
        self.nvenc_aq_spinbox = sp_aq

        self.encoder_var.trace_add("write", self._on_encoder_changed)
        self._update_nvenc_state()

    def _create_dnxhr_frame(self, parent: ttk.Frame, row: int):
        """Create the DNxHR options frame (only shown when show_extra_options=True)."""
        dnx_frame = ttk.LabelFrame(parent, text="DNxHR Split (Full-Res Dual mode only)", padding=8)
        dnx_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        dnx_frame.grid_columnconfigure(1, weight=1)

        dnx_row = 0

        chk_dnx = ttk.Checkbutton(dnx_frame, variable=self.dnxhr_fullres_split_var)
        chk_dnx.grid(row=dnx_row, column=0, sticky="w", pady=2)
        lbl_dnx = ttk.Label(dnx_frame, text="Enable DNxHR Split")
        lbl_dnx.grid(row=dnx_row, column=1, sticky="w", pady=2)
        self._add_tooltip(lbl_dnx, "dnxhr_fullres_split_tooltip")

        dnx_row += 1
        lbl_prof = ttk.Label(dnx_frame, text="DNxHR Profile:")
        lbl_prof.grid(row=dnx_row, column=0, sticky="e", padx=(0, 8), pady=2)
        self._add_tooltip(lbl_prof, "dnxhr_profile")

        cb_prof = ttk.Combobox(
            dnx_frame,
            textvariable=self.dnxhr_profile_var,
            values=("SQ (8-bit 4:2:2)", "HQ (8-bit 4:2:2)", "HQX (10-bit 4:2:2)", "444 (10-bit 4:4:4)"),
            state="readonly",
            width=20,
        )
        cb_prof.grid(row=dnx_row, column=1, sticky="w", pady=2)

        self.dnxhr_frame = dnx_frame
        self.dnxhr_profile_combo = cb_prof

        self.dnxhr_fullres_split_var.trace_add("write", self._on_dnxhr_changed)
        self._update_dnxhr_state()

    def _create_buttons(self, parent: ttk.Frame, row: int):
        """Create OK/Cancel buttons."""
        btn_row = ttk.Frame(parent)
        btn_row.grid(row=row, column=0, columnspan=2, sticky="e", pady=(15, 0))

        ttk.Button(btn_row, text="Cancel", command=self._on_cancel).pack(side="right", padx=(5, 0))
        ttk.Button(btn_row, text="OK", command=self._on_ok).pack(side="right")

    def _add_tooltip(self, widget: tk.Widget, key: str):
        """Add tooltip to widget if help text exists."""
        if key in self.help_data:
            from core.ui.widgets import Tooltip

            Tooltip(widget, self.help_data[key])

    def _on_encoder_changed(self, *args):
        """Update NVENC frame state based on encoder selection."""
        self._update_nvenc_state()

    def _on_dnxhr_changed(self, *args):
        """Update DNxHR frame state based on checkbox."""
        self._update_dnxhr_state()

    def _update_nvenc_state(self):
        """Enable/disable NVENC options based on encoder."""
        is_auto = self.encoder_var.get() == "Auto"
        state = "normal" if is_auto else "disabled"

        try:
            self.nvenc_frame.configure(state=state)
            for child in self.nvenc_frame.winfo_children():
                try:
                    child.configure(state=state)
                except tk.TclError:
                    pass
        except tk.TclError:
            pass

    def _update_dnxhr_state(self):
        """Enable/disable DNxHR profile based on checkbox."""
        is_enabled = self.dnxhr_fullres_split_var.get()
        state = "readonly" if is_enabled else "disabled"

        try:
            self.dnxhr_profile_combo.configure(state=state)
        except tk.TclError:
            pass

    def _on_ok(self):
        """Handle OK button click."""
        try:
            if self.show_dual_crf:
                crf_full = int(self.crf_full_var.get())
                crf_low = int(self.crf_low_var.get())
                if crf_full < 0 or crf_full > 51 or crf_low < 0 or crf_low > 51:
                    raise ValueError("CRF must be between 0 and 51")
            else:
                crf = int(self.crf_var.get())
                if crf < 0 or crf > 51:
                    raise ValueError("CRF must be between 0 and 51")
        except ValueError as e:
            from tkinter import messagebox

            messagebox.showerror("Invalid CRF", f"CRF must be a number between 0 and 51.\n\nError: {e}")
            return

        self.result = self.get_settings()
        self.dialog.destroy()

    def _on_cancel(self):
        """Handle Cancel button click."""
        self.result = None
        self.dialog.destroy()

    def get_settings(self) -> Dict[str, Any]:
        """Get current settings as a dictionary.

        Returns:
            Dict with all encoding settings
        """
        settings = {
            "codec": self.codec_var.get(),
            "encoding_encoder": self.encoder_var.get(),
            "encoding_quality": self.quality_var.get(),
            "encoding_tune": self.tune_var.get(),
            "nvenc_lookahead_enabled": self.nvenc_lookahead_enabled_var.get(),
            "nvenc_lookahead": self.nvenc_lookahead_var.get(),
            "nvenc_spatial_aq": self.nvenc_spatial_aq_var.get(),
            "nvenc_temporal_aq": self.nvenc_temporal_aq_var.get(),
            "nvenc_aq_strength": self.nvenc_aq_strength_var.get(),
        }

        if self.show_dual_crf:
            settings["output_crf_full"] = int(self.crf_full_var.get())
            settings["output_crf_low"] = int(self.crf_low_var.get())
        else:
            settings["output_crf"] = int(self.crf_var.get())

        if self.show_extra_options:
            settings["dnxhr_fullres_split"] = self.dnxhr_fullres_split_var.get()
            settings["dnxhr_profile"] = self.dnxhr_profile_var.get()

        if self.show_color_tags:
            settings["color_tags"] = self.color_tags_var.get()
            settings["color_tags_mode"] = self.color_tags_var.get()

        return settings

    def get_encoding_args(self, force_10bit: bool = False, crf: Optional[int] = None) -> Dict[str, Any]:
        """Get FFmpeg encoding arguments.

        Args:
            force_10bit: Whether to force 10-bit output
            crf: Optional override for CRF value

        Returns:
            Dict with codec, preset, tune, crf, pix_fmt, extra_args
        """
        if crf is None:
            if self.show_dual_crf:
                crf = int(self.crf_full_var.get())
            else:
                crf = int(self.crf_var.get())

        return build_encoder_args(
            codec=self.codec_var.get(),
            encoder=self.encoder_var.get(),
            quality=self.quality_var.get(),
            tune=self.tune_var.get(),
            crf=crf,
            force_10bit=force_10bit,
            nvenc_options={
                "lookahead_enabled": self.nvenc_lookahead_enabled_var.get(),
                "lookahead": self.nvenc_lookahead_var.get(),
                "spatial_aq": self.nvenc_spatial_aq_var.get(),
                "temporal_aq": self.nvenc_temporal_aq_var.get(),
                "aq_strength": self.nvenc_aq_strength_var.get(),
            },
        )


def create_encoding_dialog(
    parent: tk.Tk,
    app_config: Optional[Dict[str, Any]] = None,
    help_data: Optional[Dict[str, str]] = None,
    title: str = "Encoding Settings",
    show_extra_options: bool = True,
    show_dual_crf: bool = False,
    show_color_tags: bool = True,
    callback: Optional[Callable[[Optional[Dict[str, Any]]], None]] = None,
) -> Optional[EncodingSettingsDialog]:
    """Create and show an encoding settings dialog.

    Args:
        parent: Parent tkinter window
        app_config: Optional existing config dict
        help_data: Optional help text dict
        title: Dialog title
        show_extra_options: If True, show DNxHR options (for splatting).
        show_dual_crf: If True, show Full Res CRF and Low Res CRF fields.
        show_color_tags: If True, show Color Tags dropdown.
        callback: Optional callback function to call with result

    Returns:
        The dialog instance (result available via dialog.get_settings())
    """
    dialog = EncodingSettingsDialog(
        parent,
        app_config,
        help_data,
        title,
        show_extra_options=show_extra_options,
        show_dual_crf=show_dual_crf,
        show_color_tags=show_color_tags,
    )

    if callback:
        parent.wait_window(dialog.dialog)
        callback(dialog.result)

    return dialog
