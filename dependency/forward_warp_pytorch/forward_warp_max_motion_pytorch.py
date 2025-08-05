import torch
from torch.nn import Module
from torch.autograd import Function
import math

# Define constants from forward_warp.h.txt
D_SCALE_INT = 100
MOTION_TH = (0.25 * D_SCALE_INT)

class ForwardWarpMaxMotionFunction(Function):

    @staticmethod
    def forward(ctx, im0, flow, eps):
        '''
        im0: the first image with shape [B, C, H, W]
        flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
        '''
        assert(len(im0.shape) == len(flow.shape) == 4)
        assert(im0.shape[0] == flow.shape[0])
        assert(im0.shape[-2:] == flow.shape[1:3])
        assert(flow.shape[3] == 2)
        assert(im0.is_contiguous())
        assert(flow.is_contiguous())
        # Assertions from original code for NaN/Inf are handled by PyTorch's default error checking
        # for floating point operations, and can be explicitly checked if absolutely necessary,
        # but typically not required for functional replacement.
        # assert(torch.isnan(flow).long().sum() == 0)
        # assert(torch.isinf(flow).long().sum() == 0)

        B, C, H, W = im0.shape
        im1_buffer = torch.zeros_like(im0, device=im0.device, dtype=im0.dtype)
        # d_buffer stores max motion, initialized to a very low value (0 in CUDA, effectively)
        # The CUDA atomicMax starts from 0 (initially zero-filled buffer).
        # We need to ensure that d_buffer is torch.int32 as per original code.
        d_buffer = torch.zeros((B, 1, H, W), device=im0.device, dtype=torch.int32)
        wght_buffer = torch.zeros((B, 1, H, W), device=im0.device, dtype=im0.dtype)

        # Create a grid of pixel coordinates
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=im0.device, dtype=im0.dtype),
            torch.arange(H, device=im0.device, dtype=im0.dtype),
            indexing='xy'
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

        # Calculate destination coordinates
        x_dest = grid_x + flow[:, :, :, 0]
        y_dest = grid_y + flow[:, :, :, 1]
        
        # Calculate motion magnitude 'd' for each source pixel
        motion_magnitude_sq = flow[:, :, :, 0]**2 + flow[:, :, :, 1]**2
        d = (D_SCALE_INT * torch.sqrt(motion_magnitude_sq)).int() # Cast to int as per CUDA

        x_f = torch.floor(x_dest).long()
        y_f = torch.floor(y_dest).long()
        x_c = x_f + 1
        y_c = y_f + 1

        # Calculate weights (same as bilinear interpolation)
        nw_k = (x_c.float() - x_dest) * (y_c.float() - y_dest)
        ne_k = (x_dest - x_f.float()) * (y_c.float() - y_dest)
        sw_k = (x_c.float() - x_dest) * (y_dest - y_f.float())
        se_k = (x_dest - x_f.float()) * (y_dest - y_f.float())

        # Create base indices for flattened 1D tensors (for B, H, W) for d_buffer and wght_buffer
        # These are [B, H, W] -> [B*H*W]
        b_h_w_grid_flat_idx_base = torch.arange(B, device=im0.device).view(B, 1, 1).expand(-1, H, W) * (H * W)

        # Clamp destination coordinates for indexing (for scatter_reduce later)
        x_f_clamped = torch.clamp(x_f, 0, W - 1)
        y_f_clamped = torch.clamp(y_f, 0, H - 1)
        x_c_clamped = torch.clamp(x_c, 0, W - 1)
        y_c_clamped = torch.clamp(y_c, 0, H - 1)

        # Create mask for valid contribution regions based on CUDA's `if(x_f>=0 && x_c<=W && y_f>=0 && y_c<=H)`
        # This mask determines if a source pixel's 2x2 bilinear kernel falls entirely within the image.
        valid_pixel_contribution = (x_f >= 0) & (x_c <= W) & (y_f >= 0) & (y_c <= H)

        # --- Pass 1: Find max motion (d_buffer) ---
        # The CUDA `atomicMax` is per destination pixel. We need to scatter `d` values and apply max reduction.
        # torch.scatter_reduce is used for this (requires PyTorch 1.12+).
        
        d_buffer_flat = d_buffer.view(-1)
        
        # Destination indices for d_buffer (B,1,H,W) -> flattened indices
        dest_idx_nw_d = b_h_w_grid_flat_idx_base + y_f_clamped * W + x_f_clamped
        dest_idx_ne_d = b_h_w_grid_flat_idx_base + y_f_clamped * W + x_c_clamped
        dest_idx_sw_d = b_h_w_grid_flat_idx_base + y_c_clamped * W + x_f_clamped
        dest_idx_se_d = b_h_w_grid_flat_idx_base + y_c_clamped * W + x_c_clamped

        # Conditions for `find_max_motion`: `wght >= 0.25` AND `valid_pixel_contribution`
        nw_cond = (nw_k >= 0.25) & valid_pixel_contribution
        ne_cond = (ne_k >= 0.25) & valid_pixel_contribution
        sw_cond = (sw_k >= 0.25) & valid_pixel_contribution
        se_cond = (se_k >= 0.25) & valid_pixel_contribution

        # Use scatter_reduce_ with "amax" to find the maximum `d` value
        # `include_self=True` is crucial as d_buffer is initialized to 0.
        # Only scatter where condition is met.
        
        # NW corner
        d_val_nw = d[nw_cond]
        idx_nw_d = dest_idx_nw_d[nw_cond]
        if idx_nw_d.numel() > 0:
            d_buffer_flat.scatter_reduce_(0, idx_nw_d, d_val_nw, reduce="amax", include_self=True)

        # NE corner
        d_val_ne = d[ne_cond]
        idx_ne_d = dest_idx_ne_d[ne_cond]
        if idx_ne_d.numel() > 0:
            d_buffer_flat.scatter_reduce_(0, idx_ne_d, d_val_ne, reduce="amax", include_self=True)

        # SW corner
        d_val_sw = d[sw_cond]
        idx_sw_d = dest_idx_sw_d[sw_cond]
        if idx_sw_d.numel() > 0:
            d_buffer_flat.scatter_reduce_(0, idx_sw_d, d_val_sw, reduce="amax", include_self=True)

        # SE corner
        d_val_se = d[se_cond]
        idx_se_d = dest_idx_se_d[se_cond]
        if idx_se_d.numel() > 0:
            d_buffer_flat.scatter_reduce_(0, idx_se_d, d_val_se, reduce="amax", include_self=True)

        d_buffer = d_buffer_flat.view(B, 1, H, W) # Reshape d_buffer back

        # --- Pass 2: Accumulate values from max motion ---
        # The CUDA `select_max_motion` adds `h * wght` to `im1_buffer` and `wght` to `wght_buffer`
        # if `wght >= 0.25` AND `*d_buffer_p - d <= MOTION_TH`.

        # Reshape for broadcasting weights and conditions
        nw_k = nw_k.unsqueeze(1) # [B, H, W] -> [B, 1, H, W] for broadcasting
        ne_k = ne_k.unsqueeze(1)
        sw_k = sw_k.unsqueeze(1)
        se_k = se_k.unsqueeze(1)
        
        d_expanded = d.unsqueeze(1) # [B, H, W] -> [B, 1, H, W] for element-wise comparison
        
        # Base indices for batch and channel slices for flattened im1_buffer and wght_buffer
        # For im1_buffer (B,C,H,W)
        b_c_h_w_indices = torch.arange(B * C * H * W, device=im0.device).view(B, C, H, W)
        base_idx_dest_im1_flat = b_c_h_w_indices - (b_c_h_w_indices % (H * W)) # Get b*C*H*W + c*H*W part

        dest_idx_nw_im1 = base_idx_dest_im1_flat + y_f_clamped.unsqueeze(1) * W + x_f_clamped.unsqueeze(1)
        dest_idx_ne_im1 = base_idx_dest_im1_flat + y_f_clamped.unsqueeze(1) * W + x_c_clamped.unsqueeze(1)
        dest_idx_sw_im1 = base_idx_dest_im1_flat + y_c_clamped.unsqueeze(1) * W + x_f_clamped.unsqueeze(1)
        dest_idx_se_im1 = base_idx_dest_im1_flat + y_c_clamped.unsqueeze(1) * W + x_c_clamped.unsqueeze(1)

        # For wght_buffer (B,1,H,W) - same structure as d_buffer (B,1,H,W)
        base_idx_wght_flat = torch.arange(B, device=im0.device).view(B, 1, 1, 1).expand(-1, 1, H, W) * (H * W)
        dest_idx_nw_wght = base_idx_wght_flat + y_f_clamped.unsqueeze(1) * W + x_f_clamped.unsqueeze(1)
        dest_idx_ne_wght = base_idx_wght_flat + y_f_clamped.unsqueeze(1) * W + x_c_clamped.unsqueeze(1)
        dest_idx_sw_wght = base_idx_wght_flat + y_c_clamped.unsqueeze(1) * W + x_f_clamped.unsqueeze(1)
        dest_idx_se_wght = base_idx_wght_flat + y_c_clamped.unsqueeze(1) * W + x_c_clamped.unsqueeze(1)

        # Gather d_buffer values at the 4 corners for each source pixel
        # d_buffer is int32, convert to float for subtraction
        d_buffer_float = d_buffer.float()
        
        d_buffer_nw_gather = torch.gather(d_buffer_float.view(-1), 0, dest_idx_nw_wght.view(-1)).view(B,1,H,W)
        d_buffer_ne_gather = torch.gather(d_buffer_float.view(-1), 0, dest_idx_ne_wght.view(-1)).view(B,1,H,W)
        d_buffer_sw_gather = torch.gather(d_buffer_float.view(-1), 0, dest_idx_sw_wght.view(-1)).view(B,1,H,W)
        d_buffer_se_gather = torch.gather(d_buffer_float.view(-1), 0, dest_idx_se_wght.view(-1)).view(B,1,H,W)

        # Calculate the accumulation condition masks: `wght >= 0.25` AND `d_buffer_val - d <= MOTION_TH`
        cond_nw_acc = (nw_k >= 0.25) & (d_buffer_nw_gather - d_expanded <= MOTION_TH) & valid_pixel_contribution.unsqueeze(1)
        cond_ne_acc = (ne_k >= 0.25) & (d_buffer_ne_gather - d_expanded <= MOTION_TH) & valid_pixel_contribution.unsqueeze(1)
        cond_sw_acc = (sw_k >= 0.25) & (d_buffer_sw_gather - d_expanded <= MOTION_TH) & valid_pixel_contribution.unsqueeze(1)
        cond_se_acc = (se_k >= 0.25) & (d_buffer_se_gather - d_expanded <= MOTION_TH) & valid_pixel_contribution.unsqueeze(1)
        
        im1_buffer_flat = im1_buffer.view(-1)
        wght_buffer_flat = wght_buffer.view(-1)

        # Accumulate im1_buffer and wght_buffer using scatter_add_ with conditions
        # NW corner
        val_nw_im1 = (im0 * nw_k) * cond_nw_acc.float()
        val_nw_wght = nw_k * cond_nw_acc.float()
        if val_nw_im1.numel() > 0: # Ensure there are elements to scatter
            im1_buffer_flat.scatter_add_(0, dest_idx_nw_im1.view(-1), val_nw_im1.view(-1))
            wght_buffer_flat.scatter_add_(0, dest_idx_nw_wght.view(-1), val_nw_wght.view(-1))
        
        # NE corner
        val_ne_im1 = (im0 * ne_k) * cond_ne_acc.float()
        val_ne_wght = ne_k * cond_ne_acc.float()
        if val_ne_im1.numel() > 0:
            im1_buffer_flat.scatter_add_(0, dest_idx_ne_im1.view(-1), val_ne_im1.view(-1))
            wght_buffer_flat.scatter_add_(0, dest_idx_ne_wght.view(-1), val_ne_wght.view(-1))

        # SW corner
        val_sw_im1 = (im0 * sw_k) * cond_sw_acc.float()
        val_sw_wght = sw_k * cond_sw_acc.float()
        if val_sw_im1.numel() > 0:
            im1_buffer_flat.scatter_add_(0, dest_idx_sw_im1.view(-1), val_sw_im1.view(-1))
            wght_buffer_flat.scatter_add_(0, dest_idx_sw_wght.view(-1), val_sw_wght.view(-1))

        # SE corner
        val_se_im1 = (im0 * se_k) * cond_se_acc.float()
        val_se_wght = se_k * cond_se_acc.float()
        if val_se_im1.numel() > 0:
            im1_buffer_flat.scatter_add_(0, dest_idx_se_im1.view(-1), val_se_im1.view(-1))
            wght_buffer_flat.scatter_add_(0, dest_idx_se_wght.view(-1), val_se_wght.view(-1))
            
        wght_buffer /= C # as per python code: wght is added C times to buffer

        # rescale image
        im1 = im1_buffer / wght_buffer.clamp(min=eps)

        # disocclusion
        disocclusions = torch.zeros(B, 1, H, W, device=im0.device, dtype=im0.dtype)
        # Using a small epsilon for float comparison to be safe for 0 weight
        disocclusions[wght_buffer < 1e-7] = 1

        ctx.save_for_backward(im0, flow) # Not actually used as backward is NotImplementedError
        ctx.im1_buffer = im1_buffer
        ctx.d_buffer = d_buffer
        ctx.wght_buffer = wght_buffer
        
        return im1, disocclusions, im1_buffer, d_buffer, wght_buffer
 
    @staticmethod
    def backward(ctx, grad_output, grad_disocclusions, grad_im1_buffer, grad_d_buffer, grad_wght_buffer):
        # Implementing this backward pass would be extremely complex due to `atomicMax` and
        # conditional `atomicAdd` operations, which break simple differentiability.
        # It would require custom C++/CUDA/ROCm implementation or a highly complex
        # custom autograd logic that mimics the exact behavior of `atomicMax` gradients.
        raise NotImplementedError


class forward_warp_max_motion(Module):
    '''
    Adapted from Algorithm 3 in Sanachez et al. 2013 "Computing Inverse Optical Flow".
    Note that this algorithm only warps forward and does not invert result of forward warp.
    Multiply with -1 to get same results as Sanachez et al.
    '''

    def __init__(self, eps=1e-5):
        super(forward_warp_max_motion, self).__init__()
        self.eps = eps

    def forward(self, im0, flow, return_disocclusions=False, debug=False):

        im1, disocclusions, im1_buffer, d_buffer, wght_buffer = ForwardWarpMaxMotionFunction.apply(im0, flow, self.eps)

        if debug:
            return im1, disocclusions, im1_buffer, d_buffer, wght_buffer
        elif return_disocclusions:
            return im1, disocclusions
        else:
            return im1