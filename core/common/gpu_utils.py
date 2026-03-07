"""GPU and hardware utilities for StereoCrafter."""

import logging
import subprocess
import gc
import torch

logger = logging.getLogger(__name__)

CUDA_AVAILABLE = False
_CUDA_CHECK_CACHE = None

def check_cuda_availability() -> bool:
    """
    Checks if CUDA is available via PyTorch and if nvidia-smi can run.
    Sets and returns the global CUDA_AVAILABLE flag.
    """
    global CUDA_AVAILABLE, _CUDA_CHECK_CACHE
    if _CUDA_CHECK_CACHE is not None:
        return _CUDA_CHECK_CACHE

    if torch.cuda.is_available():
        logger.info("PyTorch reports CUDA is available.")
        try:
            # Further check with nvidia-smi for robustness
            subprocess.run(["nvidia-smi"], capture_output=True, check=True, timeout=2, encoding="utf-8")
            logger.debug("CUDA detected (nvidia-smi also ran successfully). NVENC can be used.")
            CUDA_AVAILABLE = True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            logger.warning(
                "nvidia-smi not found or failed. CUDA is reported by PyTorch but NVENC availability cannot be fully confirmed. Proceeding with PyTorch's report."
            )
            CUDA_AVAILABLE = True  # Rely on PyTorch if nvidia-smi not found
        except Exception as e:
            logger.error(f"Unexpected error during nvidia-smi check: {e}. Relying on PyTorch's report for CUDA.")
            CUDA_AVAILABLE = True  # Rely on PyTorch as a fallback
    else:
        logger.info("PyTorch reports CUDA is NOT available. NVENC will not be used.")
        CUDA_AVAILABLE = False

    _CUDA_CHECK_CACHE = CUDA_AVAILABLE
    return CUDA_AVAILABLE

def release_cuda_memory():
    """Releases GPU memory and performs garbage collection."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared.")
        gc.collect()
        logger.debug("Python garbage collector invoked.")
    except Exception as e:
        logger.error(f"Error releasing VRAM or during garbage collection: {e}", exc_info=True)
