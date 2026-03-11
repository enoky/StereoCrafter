"""
Base UI class with common functionality for all WebUI components
"""

import threading
import queue
import logging

logger = logging.getLogger(__name__)


class BaseWebUI:
    """
    Base class for all WebUI components providing common functionality:
    - Threading utilities (stop_event, processing_thread, message_queue)
    - Common status/progress handling
    - Shared validation methods
    """
    
    def __init__(self):
        # Processing control variables
        self.stop_event = threading.Event()
        self.processing_thread = None
        self.message_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        
    def stop_processing(self):
        """Sets the stop event to gracefully halt processing."""
        self.stop_event.set()
        return "Stopping processing...", 0
    
    def is_processing(self):
        """Check if processing is currently active."""
        return self.processing_thread is not None and self.processing_thread.is_alive()
    
    def wait_for_processing(self, timeout=None):
        """Wait for the processing thread to complete."""
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=timeout)
