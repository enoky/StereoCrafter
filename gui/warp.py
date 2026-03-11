import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

try:
    from Forward_Warp import forward_warp
    logger.info("CUDA Forward Warp is available.")
except:
    try:
        from dependency.forward_warp_pytorch import forward_warp
        logger.info("Forward Warp Pytorch is active.")
    except ImportError:
         logger.warning("Could not import forward_warp from dependency.forward_warp_pytorch. Please ensure dependency folder is in path.")
         forward_warp = None

class ForwardWarpStereo(nn.Module):
    """
    PyTorch module for forward warping an image based on a disparity map.
    """
    def __init__(self, eps=1e-6, occlu_map=False):
        super(ForwardWarpStereo, self).__init__()
        self.eps = eps
        self.occlu_map = occlu_map
        if forward_warp:
            self.fw = forward_warp()
        else:
            self.fw = None
            logger.error("forward_warp module not initialized!")

    def forward(self, im, disp):
        if self.fw is None:
             return im # Fail safe? Or raise error?
        
        im = im.contiguous()
        disp = disp.contiguous()
        weights_map = disp - disp.min()
        weights_map = (1.414) ** weights_map
        flow = -disp.squeeze(1)
        dummy_flow = torch.zeros_like(flow, requires_grad=False)
        flow = torch.stack((flow, dummy_flow), dim=-1)
        res_accum = self.fw(im * weights_map, flow)
        mask = self.fw(weights_map, flow)
        mask.clamp_(min=self.eps)
        res = res_accum / mask
        if not self.occlu_map:
            return res
        else:
            ones = torch.ones_like(disp, requires_grad=False)
            occlu_map = self.fw(ones, flow)
            occlu_map.clamp_(0.0, 1.0)
            occlu_map = 1.0 - occlu_map
            return res, occlu_map
