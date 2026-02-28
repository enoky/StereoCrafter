"""Shared utilities across all GUI applications.

This package contains modules that are common to multiple GUI applications
within the StereoCrafter project.
"""

from .video_io import VideoIO, read_video_frames

__all__ = [
    "VideoIO",
    "read_video_frames",
]
