import os
import shutil
import threading
import time
import queue
import logging
from typing import List, Tuple, Optional, Callable


def move_single_file(
    src_path: str, dest_folder: str, logger: Optional[logging.Logger] = None, max_retries: int = 1, retry_delay: float = 0.5
) -> Tuple[bool, Optional[str]]:
    """
    Moves a single file to a 'finished' subfolder within dest_folder.

    Args:
        src_path: Source file path
        dest_folder: Destination folder (finished subfolder will be created inside)
        logger: Optional logger for output
        max_retries: Number of attempts to try before failing
        retry_delay: Delay between attempts

    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    if not os.path.exists(src_path):
        return True, None  # File doesn't exist, consider it moved

    for attempt in range(max_retries):
        try:
            finished_dir = os.path.join(dest_folder, "finished")
            os.makedirs(finished_dir, exist_ok=True)

            dest_path = os.path.join(finished_dir, os.path.basename(src_path))

            if os.path.exists(dest_path):
                os.remove(src_path)
                if logger:
                    logger.info(f"Removed source file (destination exists): {os.path.basename(src_path)}")
            else:
                shutil.move(src_path, finished_dir)
                if logger:
                    logger.info(f"Moved {os.path.basename(src_path)} to {finished_dir}")

            return True, None
        except PermissionError as e:
            error_msg = f"File in use: {e}"
            if attempt < max_retries - 1:
                if logger:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: {error_msg}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            if logger:
                logger.error(f"Failed to move {os.path.basename(src_path)} after {max_retries} attempts: {error_msg}")
            return False, error_msg
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                if logger:
                    logger.warning(f"Attempt {attempt + 1}/{max_retries}: {error_msg}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                continue
            if logger:
                logger.error(f"Failed to move {os.path.basename(src_path)} after {max_retries} attempts: {error_msg}")
            return False, error_msg
    return False, "Unknown error"


def move_files_to_finished(
    files_to_move: List[Tuple[str, str]],
    logger: Optional[logging.Logger] = None,
    wait_before_move: float = 0.0,
    close_handles_callback: Optional[Callable[[], None]] = None,
    max_retries: int = 5,
    retry_delay: float = 0.5,
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """
    Synchronously moves files to 'finished' subfolders.

    Args:
        files_to_move: List of (source_path, destination_folder) tuples
        logger: Optional logger for output
        wait_before_move: Seconds to wait before moving (helps with file handle release)
        close_handles_callback: Optional callback to release file handles before moving
        max_retries: Number of attempts to try before failing
        retry_delay: Delay between attempts

    Returns:
        Tuple of (moved_count, failed_count, list of (filename, error_message) for failures)
    """
    moved_count = 0
    failed_files: List[Tuple[str, str]] = []

    # Close handles and wait if callback provided
    if close_handles_callback:
        close_handles_callback()
    if wait_before_move > 0:
        time.sleep(wait_before_move)

    for src_path, dest_folder in files_to_move:
        if not os.path.exists(src_path):
            moved_count += 1
            continue

        success, error_msg = move_single_file(src_path, dest_folder, logger, max_retries, retry_delay)

        if success:
            moved_count += 1
        else:
            failed_files.append((os.path.basename(src_path), error_msg))

    return moved_count, len(failed_files), failed_files


class FileOrganizerWorker:
    """
    Background worker that moves files to finished folders with retry logic.
    Uses a queue for async processing with automatic retries.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, max_retries: int = 3, retry_delay: float = 1.0):
        self.logger = logger
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._stats = {"moved": 0, "failed": 0, "retries": 0}

    def start(self):
        """Start the background worker thread."""
        if self._worker_thread and self._worker_thread.is_alive():
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        if self.logger:
            self.logger.debug("FileOrganizerWorker started")

    def stop(self, wait: bool = True, timeout: float = 5.0):
        """Stop the background worker."""
        self._stop_event.set()

        if wait and self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=timeout)

        if self.logger:
            self.logger.debug(
                f"FileOrganizerWorker stopped. Stats: moved={self._stats['moved']}, "
                f"failed={self._stats['failed']}, retries={self._stats['retries']}"
            )

    def put(self, src_path: str, dest_folder: str, priority: int = 0):
        """
        Add a file to the move queue.

        Args:
            src_path: Source file path
            dest_folder: Destination folder (finished subfolder will be created inside)
            priority: Higher priority items are processed first (default 0)
        """
        self._queue.put((priority, src_path, dest_folder))

    def put_batch(self, files: List[Tuple[str, str]], priority: int = 0):
        """Add multiple files to the queue."""
        for src_path, dest_folder in files:
            self.put(src_path, dest_folder, priority)

    def wait_until_empty(self, timeout: float = 30.0):
        """Wait until the queue is empty."""
        start_time = time.time()
        while not self._queue.empty():
            if time.time() - start_time > timeout:
                if self.logger:
                    self.logger.warning("Timeout waiting for file organizer queue to empty")
                return
            time.sleep(0.1)

    def get_stats(self) -> dict:
        """Get current statistics."""
        return self._stats.copy()

    def _worker_loop(self):
        """Main worker loop that processes the queue."""
        while not self._stop_event.is_set():
            try:
                # Get item with timeout so we can check stop_event
                priority, src_path, dest_folder = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break

            # Try to move the file with retries
            success = False
            for attempt in range(self.max_retries):
                result, error_msg = move_single_file(src_path, dest_folder, logger=self.logger, max_retries=1)

                if result:
                    success = True
                    self._stats["moved"] += 1
                    if self.logger:
                        self.logger.debug(f"Moved to finished: {os.path.basename(src_path)}")
                    break
                else:
                    self._stats["retries"] += 1
                    if attempt < self.max_retries - 1:
                        if self.logger:
                            self.logger.debug(
                                f"Retry {attempt + 1}/{self.max_retries} for {os.path.basename(src_path)}: {error_msg}"
                            )
                        time.sleep(self.retry_delay)

            if not success:
                self._stats["failed"] += 1
                if self.logger:
                    self.logger.error(f"Failed to move after {self.max_retries} attempts: {os.path.basename(src_path)}")

            self._queue.task_done()


def restore_finished_files(
    restore_dirs: List[Tuple[str, str]],
    logger: Optional[logging.Logger] = None,
    wait_before_move: float = 0.0,
    close_handles_callback: Optional[Callable[[], None]] = None,
) -> Tuple[int, int, List[Tuple[str, str]]]:
    """
    Moves files from 'finished' subfolders back to their original input directories.

    Args:
        restore_dirs: List of (input_folder, finished_folder) tuples
        logger: Optional logger for output
        wait_before_move: Seconds to wait before moving
        close_handles_callback: Optional callback to release file handles before moving

    Returns:
        Tuple of (restored_count, failed_count, list of (filename, error_message) for failures)
    """
    restored_count = 0
    failed_files: List[Tuple[str, str]] = []

    if close_handles_callback:
        close_handles_callback()
    if wait_before_move > 0:
        time.sleep(wait_before_move)

    for input_folder, finished_folder in restore_dirs:
        finished_path = os.path.join(input_folder, finished_folder)

        if not os.path.isdir(finished_path):
            if logger:
                logger.debug(f"Restore skipped: 'finished' folder not found at {finished_path}")
            continue

        try:
            files_to_restore = [f for f in os.listdir(finished_path) if os.path.isfile(os.path.join(finished_path, f))]
        except Exception as e:
            if logger:
                logger.error(f"Error listing finished folder {finished_path}: {e}")
            failed_files.append((finished_folder, str(e)))
            continue

        if not files_to_restore:
            if logger:
                logger.debug(f"Restore skipped: No files found in {finished_path}")
            continue

        for filename in files_to_restore:
            src_path = os.path.join(finished_path, filename)
            dest_path = os.path.join(input_folder, filename)

            for attempt in range(5):
                try:
                    if os.path.exists(dest_path):
                        os.remove(src_path)
                        if logger:
                            logger.info(f"Removed duplicate from finished: {filename}")
                    else:
                        shutil.move(src_path, input_folder)
                        if logger:
                            logger.info(f"Restored '{filename}' from finished to {input_folder}")
                    restored_count += 1
                    break
                except Exception as e:
                    error_msg = str(e)
                    if attempt < 4:
                        if logger:
                            logger.warning(f"Error restoring '{filename}': {error_msg}. Retrying in 0.5s...")
                        time.sleep(0.5)
                        continue
                    if logger:
                        logger.error(f"Error restoring '{filename}' after 5 attempts: {error_msg}")
                    failed_files.append((filename, error_msg))
                    break

        # Remove empty finished folder
        try:
            if os.path.exists(finished_path) and not os.listdir(finished_path):
                os.rmdir(finished_path)
                if logger:
                    logger.debug(f"Removed empty finished folder: {finished_path}")
        except Exception as e:
            if logger:
                logger.warning(f"Could not remove empty finished folder '{finished_path}': {e}")

    return restored_count, len(failed_files), failed_files
