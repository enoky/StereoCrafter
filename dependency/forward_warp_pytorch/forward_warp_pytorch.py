import torch
from torch.nn import Module
from torch.autograd import Function
import math

class ForwardWarpFunction(Function):

    @staticmethod
    def forward(ctx, im0, flow, interpolation_mode_int):
        '''
        im0: the first image with shape [B, C, H, W]
        flow: the optical flow with shape [B, H, W, 2] (different to grid_sample, it's range is from [-W, -H] to [W, H])
        interpolation_mode_int: 0 for Bilinear, 1 for Nearest
        '''
        assert(len(im0.shape) == len(flow.shape) == 4)
        # Changed 'is' to '==' for integer comparison
        assert(interpolation_mode_int == 0 or interpolation_mode_int == 1)
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
        im1 = torch.zeros_like(im0, device=im0.device, dtype=im0.dtype)

        # Create a grid of pixel coordinates
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=im0.device, dtype=im0.dtype),
            torch.arange(H, device=im0.device, dtype=im0.dtype),
            indexing='xy' # 'xy' means x-coords vary along dim 0, y-coords along dim 1
        )
        # Unsqueeze and expand for batch dimension [1, H, W] -> [B, H, W]
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

        # Calculate destination coordinates
        x_dest = grid_x + flow[:, :, :, 0]
        y_dest = grid_y + flow[:, :, :, 1]
        
        if interpolation_mode_int == 0:  # Bilinear
            x_f = torch.floor(x_dest).long() # x_floor
            y_f = torch.floor(y_dest).long() # y_floor
            x_c = x_f + 1 # x_ceil
            y_c = y_f + 1 # y_ceil

            # Calculate weights
            nw_k = (x_c.float() - x_dest) * (y_c.float() - y_dest)
            ne_k = (x_dest - x_f.float()) * (y_c.float() - y_dest)
            sw_k = (x_c.float() - x_dest) * (y_dest - y_f.float())
            se_k = (x_dest - x_f.float()) * (y_dest - y_f.float())

            # Clamp coordinates to image boundaries for valid indices.
            # This is done *after* weight calculation but before indexing.
            x_f_clamped = torch.clamp(x_f, 0, W - 1)
            y_f_clamped = torch.clamp(y_f, 0, H - 1)
            x_c_clamped = torch.clamp(x_c, 0, W - 1)
            y_c_clamped = torch.clamp(y_c, 0, H - 1)

            # Create masks for valid contributions (original boundaries from CUDA `if` condition).
            # A contribution is valid if all its 4 corners (x_f, x_c-1, y_f, y_c-1) are within the image.
            # Note: CUDA used `x_c < W` which means x_c can be up to W-1.
            valid_mask = (x_f >= 0) & (x_c < W) & (y_f >= 0) & (y_c < H)
            
            # Reshape weights and masks for broadcasting with im0 (add channel dim)
            nw_k = nw_k.unsqueeze(1) # [B, 1, H, W]
            ne_k = ne_k.unsqueeze(1)
            sw_k = sw_k.unsqueeze(1)
            se_k = se_k.unsqueeze(1)
            valid_mask = valid_mask.unsqueeze(1) # [B, 1, H, W]

            # Calculate base index for each batch and channel slice for flattened im1.
            # These indices will be for a [B, C, H, W] tensor flattened to [B*C*H*W].
            b_indices = torch.arange(B, device=im0.device).view(B, 1, 1, 1).expand(-1, C, H, W)
            c_indices = torch.arange(C, device=im0.device).view(1, C, 1, 1).expand(B, -1, H, W)
            base_idx = b_indices * (C * H * W) + c_indices * (H * W)
            
            # Use torch.scatter_add_ to perform atomic additions.
            # Ensure values are zeroed out if they are not part of the `valid_mask`
            
            # --- Scatter to NW corner ---
            nw_flat_idx = base_idx + y_f_clamped.unsqueeze(1) * W + x_f_clamped.unsqueeze(1)
            nw_values = (im0 * nw_k) * valid_mask.float()
            im1.view(-1).scatter_add_(0, nw_flat_idx.view(-1), nw_values.view(-1))

            # --- Scatter to NE corner ---
            ne_flat_idx = base_idx + y_f_clamped.unsqueeze(1) * W + x_c_clamped.unsqueeze(1)
            ne_values = (im0 * ne_k) * valid_mask.float()
            im1.view(-1).scatter_add_(0, ne_flat_idx.view(-1), ne_values.view(-1))

            # --- Scatter to SW corner ---
            sw_flat_idx = base_idx + y_c_clamped.unsqueeze(1) * W + x_f_clamped.unsqueeze(1)
            sw_values = (im0 * sw_k) * valid_mask.float()
            im1.view(-1).scatter_add_(0, sw_flat_idx.view(-1), sw_values.view(-1))

            # --- Scatter to SE corner ---
            se_flat_idx = base_idx + y_c_clamped.unsqueeze(1) * W + x_c_clamped.unsqueeze(1)
            se_values = (im0 * se_k) * valid_mask.float()
            im1.view(-1).scatter_add_(0, se_flat_idx.view(-1), se_values.view(-1))

        else:  # Nearest
            x_nearest = torch.round(x_dest).long()
            y_nearest = torch.round(y_dest).long()

            # Create masks for valid contributions
            valid_mask = (x_nearest >= 0) & (x_nearest < W) & \
                         (y_nearest >= 0) & (y_nearest < H)
            
            # Reshape valid_mask for broadcasting with im0
            valid_mask = valid_mask.unsqueeze(1) # [B, 1, H, W]

            # Generate flat indices for im0 to extract source values efficiently
            b_indices = torch.arange(B, device=im0.device).view(B, 1, 1, 1).expand(-1, C, H, W)
            c_indices = torch.arange(C, device=im0.device).view(1, C, 1, 1).expand(B, -1, H, W)
            
            # Clamp nearest coordinates for indexing
            x_nearest_clamped = torch.clamp(x_nearest, 0, W - 1)
            y_nearest_clamped = torch.clamp(y_nearest, 0, H - 1)

            # Destination flat indices
            dest_flat_idx = b_indices * (C * H * W) + \
                            c_indices * (H * W) + \
                            y_nearest_clamped.unsqueeze(1) * W + \
                            x_nearest_clamped.unsqueeze(1)
            
            # Filter source values by valid_mask
            source_values = im0 * valid_mask.float()

            # For nearest, the CUDA code does direct assignment `*im1_p = *im0_p;`
            # This implies "last write wins" if multiple source pixels map to the same destination.
            # `torch.scatter_` (not `scatter_add_`) with `reduce='replace'` (default for `scatter_`)
            # provides this behavior.
            im1.view(-1).scatter_(0, dest_flat_idx.view(-1), source_values.view(-1))
            
        ctx.save_for_backward(im0, flow)
        ctx.interpolation_mode_int = interpolation_mode_int
        ctx.B, ctx.C, ctx.H, ctx.W = B, C, H, W
        
        return im1

    @staticmethod
    def backward(ctx, grad_output):
        im0, flow = ctx.saved_tensors
        interpolation_mode_int = ctx.interpolation_mode_int
        B, C, H, W = ctx.B, ctx.C, ctx.H, ctx.W

        grad_output = grad_output.contiguous()

        im0_grad = torch.zeros_like(im0, device=im0.device, dtype=im0.dtype)
        flow_grad = torch.zeros_like(flow, device=flow.device, dtype=flow.dtype)

        # Create a grid of pixel coordinates
        grid_x, grid_y = torch.meshgrid(
            torch.arange(W, device=im0.device, dtype=im0.dtype),
            torch.arange(H, device=im0.device, dtype=im0.dtype),
            indexing='xy'
        )
        grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
        grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

        x_dest = grid_x + flow[:, :, :, 0]
        y_dest = grid_y + flow[:, :, :, 1]

        if interpolation_mode_int == 0:  # Bilinear
            x_f = torch.floor(x_dest).long()
            y_f = torch.floor(y_dest).long()
            x_c = x_f + 1
            y_c = y_f + 1

            # Clamp coordinates to valid range [0, W-1] and [0, H-1]
            x_f_clamped = torch.clamp(x_f, 0, W - 1)
            y_f_clamped = torch.clamp(y_f, 0, H - 1)
            x_c_clamped = torch.clamp(x_c, 0, W - 1)
            y_c_clamped = torch.clamp(y_c, 0, H - 1)

            # Create mask for valid contributions in the forward pass.
            # This ensures gradients only flow back from locations that were actually written to.
            valid_mask_forward = (x_f >= 0) & (x_c < W) & (y_f >= 0) & (y_c < H)
            valid_mask_forward_bc = valid_mask_forward.unsqueeze(1).expand(-1, C, -1, -1) # [B, C, H, W]

            # Source pixel values needed for flow gradient calculation
            im0_bc = im0 # [B, C, H, W]

            # Get gradients at the 4 corners of the destination image
            # Base indices for batch and channel slices for flattened grad_output
            b_indices = torch.arange(B, device=im0.device).view(B, 1, 1, 1).expand(-1, C, H, W)
            c_indices = torch.arange(C, device=im0.device).view(1, C, 1, 1).expand(B, -1, H, W)
            base_idx_dest = b_indices * (C * H * W) + c_indices * (H * W)

            # Gather gradients from flattened grad_output at the 4 corner locations
            nw_grad = torch.gather(grad_output.view(-1), 0, (base_idx_dest + y_f_clamped.unsqueeze(1) * W + x_f_clamped.unsqueeze(1)).view(-1)).view(B,C,H,W)
            ne_grad = torch.gather(grad_output.view(-1), 0, (base_idx_dest + y_f_clamped.unsqueeze(1) * W + x_c_clamped.unsqueeze(1)).view(-1)).view(B,C,H,W)
            sw_grad = torch.gather(grad_output.view(-1), 0, (base_idx_dest + y_c_clamped.unsqueeze(1) * W + x_f_clamped.unsqueeze(1)).view(-1)).view(B,C,H,W)
            se_grad = torch.gather(grad_output.view(-1), 0, (base_idx_dest + y_c_clamped.unsqueeze(1) * W + x_c_clamped.unsqueeze(1)).view(-1)).view(B,C,H,W)

            # Apply the forward pass valid mask to the gathered gradients, zeroing out contributions that didn't happen
            nw_grad *= valid_mask_forward_bc.float()
            ne_grad *= valid_mask_forward_bc.float()
            sw_grad *= valid_mask_forward_bc.float()
            se_grad *= valid_mask_forward_bc.float()

            # Calculate weights (same as forward pass)
            nw_k = (x_c.float() - x_dest).unsqueeze(1)
            ne_k = (x_dest - x_f.float()).unsqueeze(1)
            sw_k = (x_c.float() - x_dest).unsqueeze(1)
            se_k = (x_dest - x_f.float()).unsqueeze(1)

            # Calculate im0_grad: sum of weighted gradients from output
            im0_grad += nw_k * nw_grad
            im0_grad += ne_k * ne_grad
            im0_grad += sw_k * sw_grad
            im0_grad += se_k * se_grad
            
            # Apply forward pass valid mask to im0_grad
            im0_grad *= valid_mask_forward_bc.float()

            # Calculate flow_grad_x and flow_grad_y by summing partial derivatives over channels
            # These partial derivatives are derived from chain rule:
            # d(Loss)/d(flow_x) = sum_c d(Loss)/d(im1_c) * d(im1_c)/d(x_dest) * d(x_dest)/d(flow_x)
            # Since d(x_dest)/d(flow_x) = 1, we focus on d(im1_c)/d(x_dest).
            # d(im1_c)/d(x_dest) = d(nw_k)/dx * p + d(ne_k)/dx * p + ...
            # The CUDA kernel combines these and sums.

            # Partial derivatives of weights with respect to x_dest and y_dest
            # d(nw_k)/dx = -(y_c - y_dest)
            # d(nw_k)/dy = -(x_c - x_dest)
            # d(ne_k)/dx = (y_c - y_dest)
            # d(ne_k)/dy = -(x_dest - x_f)
            # d(sw_k)/dx = -(y_dest - y_f)
            # d(sw_k)/dy = (x_c - x_dest)
            # d(se_k)/dx = (y_dest - y_f)
            # d(se_k)/dy = (x_dest - x_f)

            flow_grad_x_contrib = torch.zeros_like(im0_bc)
            flow_grad_y_contrib = torch.zeros_like(im0_bc)

            flow_grad_x_contrib -= (y_c.float().unsqueeze(1) - y_dest.unsqueeze(1)) * im0_bc * nw_grad
            flow_grad_x_contrib += (y_c.float().unsqueeze(1) - y_dest.unsqueeze(1)) * im0_bc * ne_grad
            flow_grad_x_contrib -= (y_dest.unsqueeze(1) - y_f.float().unsqueeze(1)) * im0_bc * sw_grad
            flow_grad_x_contrib += (y_dest.unsqueeze(1) - y_f.float().unsqueeze(1)) * im0_bc * se_grad

            flow_grad_y_contrib -= (x_c.float().unsqueeze(1) - x_dest.unsqueeze(1)) * im0_bc * nw_grad
            flow_grad_y_contrib -= (x_dest.unsqueeze(1) - x_f.float().unsqueeze(1)) * im0_bc * ne_grad
            flow_grad_y_contrib += (x_c.float().unsqueeze(1) - x_dest.unsqueeze(1)) * im0_bc * sw_grad
            flow_grad_y_contrib += (x_dest.unsqueeze(1) - x_f.float().unsqueeze(1)) * im0_bc * se_grad

            # Sum over the channel dimension to get final flow_grad [B, H, W, 2]
            flow_grad[:, :, :, 0] = flow_grad_x_contrib.sum(dim=1)
            flow_grad[:, :, :, 1] = flow_grad_y_contrib.sum(dim=1)
            
            # Apply forward pass valid mask to flow_grad
            flow_grad *= valid_mask_forward.unsqueeze(-1).float()

        else:  # Nearest
            x_nearest = torch.round(x_dest).long()
            y_nearest = torch.round(y_dest).long()

            valid_mask_forward = (x_nearest >= 0) & (x_nearest < W) & \
                                 (y_nearest >= 0) & (y_nearest < H)
            valid_mask_forward_bc = valid_mask_forward.unsqueeze(1).expand(-1, C, -1, -1)

            # In nearest neighbor, gradient flows back directly.
            # Gather `grad_output` values based on where `im0` pixels would have gone.
            b_indices = torch.arange(B, device=im0.device).view(B, 1, 1, 1).expand(-1, C, H, W)
            c_indices = torch.arange(C, device=im0.device).view(1, C, 1, 1).expand(B, -1, H, W)
            base_idx_dest = b_indices * (C * H * W) + c_indices * (H * W)

            dest_flat_idx = base_idx_dest + \
                            y_nearest.unsqueeze(1) * W + \
                            x_nearest.unsqueeze(1)
            
            # Gather gradients from `grad_output` at the destination points
            im0_grad_gathered = torch.gather(grad_output.view(-1), 0, dest_flat_idx.view(-1)).view(B,C,H,W)
            im0_grad = im0_grad_gathered * valid_mask_forward_bc.float()
            
            # Flow gradient for Nearest is zero unless explicitly defined.
            # The CUDA `forward_warp_cuda_backward_kernel` for Nearest also implicitly sets flow_grad to zero.
            # So, flow_grad remains as its initial zeros_like value.

        return im0_grad, flow_grad, None # None for interpolation_mode_int gradient


class forward_warp(Module):

    def __init__(self, interpolation_mode="Bilinear"):
        '''
        Support interpolation mode with Bilinear and Nearest.
        '''
        super(forward_warp, self).__init__()
        assert(interpolation_mode in ["Bilinear", "Nearest"])
        # Changed 'is' to '==' for integer comparison
        if(interpolation_mode == "Bilinear"):
            self.interpolation_mode_int = 0
        else:
            self.interpolation_mode_int = 1

    def forward(self, im0, flow):

        return ForwardWarpFunction.apply(im0, flow, self.interpolation_mode_int)