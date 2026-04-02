"""Drag-and-drop helpers for path Entry widgets (tkinterdnd2 / tkdnd).

Usage
-----
1.  Call ``init_dnd(root)`` once after your ``ThemedTk.__init__()`` finishes.
    It loads the native tkdnd library and returns ``True`` if DnD is available.

2.  Call ``register_dnd_entries(entries)`` to make a list of
    ``(ttk.Entry, tk.StringVar)`` tuples accept file/folder drops.

3.  Call ``configure_dnd_styles(style, is_dark)`` inside your ``_apply_theme``
    to keep the highlight colour in sync with the current theme.

All functions degrade gracefully if tkinterdnd2 is not installed — no crash,
no visual change, no error.
"""

import os
import re
import logging

logger = logging.getLogger(__name__)

# --- Optional import ----------------------------------------------------------
_DND_AVAILABLE = False
try:
    from tkinterdnd2 import DND_FILES
    from tkinterdnd2.TkinterDnD import _require as _dnd_require

    _DND_AVAILABLE = True
except ImportError:
    pass


# --- Public API ---------------------------------------------------------------


def init_dnd(root) -> bool:
    """Load the tkdnd Tcl package into *root* (a Tk or ThemedTk window).

    Returns ``True`` if drag-and-drop is now functional, ``False`` otherwise.
    Safe to call even when tkinterdnd2 is not installed.
    """
    if not _DND_AVAILABLE:
        logger.info("tkinterdnd2 not installed — drag-and-drop disabled.")
        return False
    try:
        _dnd_require(root)
        logger.debug("Drag-and-drop support enabled.")
        return True
    except Exception as e:
        logger.debug(f"Drag-and-drop not available: {e}")
        return False


def register_dnd_entries(entries, dnd_enabled: bool = True):
    """Register a list of path entries as drop targets.

    *entries* is a list of tuples:
    ``(entry_widget, tk_var, folder_only, allowed_extensions)``

    - ``folder_only``: If True, and a file is dropped, the parent folder is used.
    - ``allowed_extensions``: A list/tuple of extensions (e.g. ['.mp4', '.mkv'])
      to filter file drops. Ignored for folders.
    """
    if not dnd_enabled or not _DND_AVAILABLE:
        return
    for entry_data in entries:
        # Support both (entry, var) and (entry, var, folder_only, extensions)
        if len(entry_data) == 2:
            entry, var = entry_data
            folder_only = False
            extensions = None
        else:
            entry, var, folder_only, extensions = entry_data

        _register_single(entry, var, folder_only, extensions)
    logger.info(f"Drag-and-drop registered on {len(entries)} path entries.")


def configure_dnd_styles(style, is_dark: bool, dnd_enabled: bool = True):
    """Create / update the ``DropHighlight.TEntry`` ttk style.

    Call this from your ``_apply_theme`` method so the highlight tracks
    the current dark/light mode.
    """
    if not dnd_enabled:
        return
    if is_dark:
        style.configure("DropHighlight.TEntry", fieldbackground="#264f78")
    else:
        style.configure("DropHighlight.TEntry", fieldbackground="#cce8ff")


# --- Internal -----------------------------------------------------------------


def _register_single(entry, var, folder_only=False, extensions=None):
    """Wire up DnD events on a single ``ttk.Entry`` widget."""
    try:
        entry.drop_target_register(DND_FILES)
    except Exception as e:
        logger.warning(f"DnD registration failed for {entry._w}: {e}")
        return

    def on_drop(event):
        path = _parse_dnd_data(event.data)
        if not path:
            return event.action

        is_dir = os.path.isdir(path)

        if folder_only:
            # If it's a file, take the directory it lives in
            if not is_dir:
                path = os.path.dirname(path)
            var.set(path)
        else:
            # File filtering (if extensions provided)
            if not is_dir and extensions:
                ext = os.path.splitext(path)[1].lower()
                if ext not in [e.lower() for e in extensions]:
                    logger.warning(f"Dropped file hidden: {ext} not in {extensions}")
                    # Visual feedback: reset highlight but don't set var
                    entry.configure(style="TEntry")
                    return event.action
            var.set(path)

        try:
            entry.configure(style="TEntry")
        except Exception:
            pass
        return event.action

    def on_enter(event):
        try:
            entry.configure(style="DropHighlight.TEntry")
        except Exception:
            pass
        return event.action

    def on_position(event):
        return event.action

    def on_leave(event):
        try:
            entry.configure(style="TEntry")
        except Exception:
            pass

    entry.dnd_bind("<<Drop>>", on_drop)
    entry.dnd_bind("<<DropEnter>>", on_enter)
    entry.dnd_bind("<<DropPosition>>", on_position)
    entry.dnd_bind("<<DropLeave>>", on_leave)


def _parse_dnd_data(data):
    """Parse tkdnd dropped-data string into a single normalised path."""
    if not data:
        return None
    data = data.strip()
    if not data:
        return None
    # tkdnd wraps paths containing spaces in curly braces
    if data.startswith("{"):
        paths = re.findall(r"\{([^}]+)\}", data)
    else:
        paths = data.split()
    if not paths:
        return None
    return os.path.normpath(paths[0])
