"""Forward warp module for stereo video generation.

Provides the ForwardWarpStereo PyTorch module for warping images
based on disparity maps for 3D video generation.
"""

import logging
from typing import Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from Forward_Warp import forward_warp

    logger.info("CUDA Forward Warp is available.")
except Exception:
    from dependency.forward_warp_pytorch import forward_warp

    logger.info("Forward Warp Pytorch is active.")


class ForwardWarpStereo(nn.Module):
    """
    PyTorch module for forward warping an image based on a disparity map.

    This module performs forward warping where each pixel in the input
    image is shifted according to the disparity map to create a warped
    output suitable for stereoscopic 3D video generation.

    Args:
        eps: Small epsilon value for numerical stability (default: 1e-6)
        occlu_map: Whether to return occlusion map alongside warped image
                   (default: False)

    Example:
        >>> import torch
        >>> from core.splatting.forward_warp import ForwardWarpStereo
        >>> 
        >>> # Create module
        >>> fws = ForwardWarpStereo(eps=1e-6, occlu_map=False)
        >>> 
        >>> # Forward pass
        >>> image = torch.randn(1, 3, 512, 1024)  # [B, C, H, W]
        >>> disparity = torch.randn(1, 1, 512, 512)  # [B, 1, H, W]
        >>> warped = fws(image, disparity)
    """

    def __init__(self, eps: float = 1e-6, occlu_map: bool = False):
        """Initialize the forward warp stereo module.

        Args:
            eps: Small value to prevent division by zero
            occlu_map: Whether to return occlusion map
        """
        super().__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        self.fw = forward_warp()

    def forward(
        self,
        im: torch.Tensor,
        disp: torch.Tensor,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Perform forward warping of image based on disparity.

        Args:
            im: Input image tensor of shape [B, C, H, W]
            disp: Disparity map tensor of shape [B, 1, H, W]
                  Positive values shift pixels right (right eye view)

        Returns:
            If occlu_map is False:
                Warped image tensor of shape [B, C, H, W]
            If occlu_map is True:
                Tuple of (warped_image, occlusion_map)
                - warped_image: [B, C, H, W]
                - occlusion_map: [B, 1, H, W] showing areas without source pixels
        """
        im = im.contiguous()
        disp = disp.contiguous()

        # Create weights map from disparity
        weights_map = disp - disp.min()
        weights_map = (1.414) ** weights_map

        # Create flow field from disparity (negative for right eye shift)
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)

        # Perform forward warp
        res_accum = self.fw(im * weights_map, flow)
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask

        if not self.occlu_map:
            return res
        else:
            # Compute occlusion map
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map
