import logging

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    # Fallback/stub for systems without moviepy
    class VideoFileClip:
        def __init__(self, *args, **kwargs):
            logging.warning("moviepy.editor not found. Frame counting disabled.")
        def close(self): pass
        @property
        def fps(self): return None
        @property
        def duration(self): return None
