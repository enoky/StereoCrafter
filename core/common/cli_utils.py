"""CLI and console utilities for StereoCrafter."""

import sys
import logging

def draw_progress_bar(current, total, bar_length=50, prefix="Progress:", suffix=""):
    """
    Draws an ASCII progress bar in the console, overwriting the same line.
    Adds a newline only when 100% complete.
    """
    if total == 0:
        sys.stdout.write(f"\r{prefix} [Skipped (Total 0)] {suffix}")
        sys.stdout.flush()
        return

    percent = 100 * (current / float(total))
    filled_length = int(round(bar_length * current / float(total)))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)

    # Format the suffix for completion
    actual_suffix = suffix
    if current == total:
        actual_suffix = "Complete      "

    sys.stdout.write(f"\r{prefix} |{bar}| {percent:.1f}% {actual_suffix}")
    sys.stdout.flush()

    if current == total:
        sys.stdout.write("\n")
        sys.stdout.flush()


def set_logger_level(logger, level):
    """Set the logging level for a logger and all its handlers.

    Args:
        logger: The logger instance to modify
        level: The logging level to set (e.g., logging.INFO)
    """
    logger.setLevel(level)
    for h in logger.handlers:
        h.setLevel(level)
