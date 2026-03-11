"""
StereoCrafter UI Package
A modular interface for depth estimation, splatting, inpainting, and merging operations.
"""

from .depthcrafter.depthcrafter_ui import DepthCrafterWebUI
from .splatting.splatting_ui import SplatterWebUI
from .inpainting.inpainting_ui import InpaintingWebUI
from .merging.merging_ui import MergingWebUI

__all__ = [
    'DepthCrafterWebUI',
    'SplatterWebUI',
    'InpaintingWebUI',
    'MergingWebUI',
]
