"""Image and mask processing utilities for StereoCrafter.

Provides GPU-accelerated and CPU-fallback functions for common image operations
like dilation, blurring, shadow effects, and anaglyph transformations.
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def apply_mask_dilation(mask: torch.Tensor, kernel_size: int, use_gpu: bool = True) -> torch.Tensor:
    """Applies dilation to a mask tensor.

    Args:
        mask: Input tensor [B, C, H, W] or [C, H, W]
        kernel_size: Size of the dilation kernel
        use_gpu: Whether to use GPU acceleration

    Returns:
        Dilated mask tensor
    """
    if kernel_size <= 0:
        return mask
    
    # Ensure 4D
    added_batch = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
        added_batch = True
        
    kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if use_gpu and mask.is_cuda:
        padding = kernel_val // 2
        result = F.max_pool2d(mask, kernel_size=kernel_val, stride=1, padding=padding)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_val, kernel_val))
        processed_frames = []
        # Expecting [B, C, H, W]
        for t in range(mask.shape[0]):
            # Squeeze to [H, W] if C=1
            frame_np = (mask[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            dilated_np = cv2.dilate(frame_np, kernel, iterations=1)
            dilated_tensor = torch.from_numpy(dilated_np).float() / 255.0
            processed_frames.append(dilated_tensor.unsqueeze(0)) # Add C dimension back
        result = torch.stack(processed_frames).to(mask.device)
        
    if added_batch:
        result = result.squeeze(0)
    return result


def apply_gaussian_blur(mask: torch.Tensor, kernel_size: int, use_gpu: bool = True) -> torch.Tensor:
    """Applies Gaussian blur to a mask tensor.

    Args:
        mask: Input tensor [B, C, H, W] or [C, H, W]
        kernel_size: Size of the blur kernel
        use_gpu: Whether to use GPU acceleration

    Returns:
        Blurred mask tensor
    """
    if kernel_size <= 0:
        return mask
        
    # Ensure 4D
    added_batch = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
        added_batch = True
        
    kernel_val = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    if use_gpu and mask.is_cuda:
        sigma = kernel_val / 6.0
        ax = torch.arange(-kernel_val // 2 + 1.0, kernel_val // 2 + 1.0, device=mask.device)
        gauss = torch.exp(-(ax**2) / (2 * sigma**2))
        kernel_1d = (gauss / gauss.sum()).view(1, 1, 1, kernel_val)
        # Apply separable convolution
        blurred_mask = F.conv2d(mask, kernel_1d, padding=(0, kernel_val // 2), groups=mask.shape[1])
        blurred_mask = F.conv2d(
            blurred_mask, kernel_1d.permute(0, 1, 3, 2), padding=(kernel_val // 2, 0), groups=mask.shape[1]
        )
        result = torch.clamp(blurred_mask, 0.0, 1.0)
    else:
        processed_frames = []
        for t in range(mask.shape[0]):
            frame_np = (mask[t].squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            blurred_np = cv2.GaussianBlur(frame_np, (kernel_val, kernel_val), 0)
            blurred_tensor = torch.from_numpy(blurred_np).float() / 255.0
            processed_frames.append(blurred_tensor.unsqueeze(0))
        result = torch.stack(processed_frames).to(mask.device)
        
    if added_batch:
        result = result.squeeze(0)
    return result


def apply_shadow_blur(
    mask: torch.Tensor,
    shift_per_step: int,
    start_opacity: float,
    opacity_decay_per_step: float,
    min_opacity: float,
    decay_gamma: float = 1.0,
    use_gpu: bool = True,
) -> torch.Tensor:
    """Applies a directional shadow blur effect to a mask.

    Args:
        mask: Input mask [B, C, H, W] or [C, H, W]
        shift_per_step: Pixel shift for each step
        start_opacity: Initial shadow opacity
        opacity_decay_per_step: Opacity reduction per step
        min_opacity: Minimum shadow opacity
        decay_gamma: Gamma correction for decay curve
        use_gpu: Whether to use GPU acceleration

    Returns:
        Mask with shadow effect applied
    """
    if shift_per_step <= 0:
        return mask
        
    # Prevent division by zero if opacity decay is zero
    if opacity_decay_per_step <= 1e-6:
        return mask
        
    num_steps = int((start_opacity - min_opacity) / opacity_decay_per_step) + 1
    if num_steps <= 0:
        return mask

    # Ensure 4D
    added_batch = False
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
        added_batch = True

    if use_gpu and mask.is_cuda:
        canvas_mask = mask.clone()
        stamp_source = mask.clone()
        for i in range(num_steps):
            t = 1.0 - (i / (num_steps - 1)) if num_steps > 1 else 1.0
            curved_t = t**decay_gamma
            current_opacity = min_opacity + (start_opacity - min_opacity) * curved_t
            total_shift = (i + 1) * shift_per_step
            padded_stamp = F.pad(stamp_source, (total_shift, 0), "constant", 0)
            shifted_stamp = padded_stamp[:, :, :, :-total_shift]
            canvas_mask = torch.max(canvas_mask, shifted_stamp * current_opacity)
        result = canvas_mask
    else:
        processed_frames = []
        for t in range(mask.shape[0]):
            canvas_np = mask[t].squeeze(0).cpu().numpy()  # Process one frame at a time
            stamp_source_np = canvas_np.copy()
            for i in range(num_steps):
                time_step = 1.0 - (i / (num_steps - 1)) if num_steps > 1 else 1.0
                curved_t = time_step**decay_gamma
                current_opacity = min_opacity + (start_opacity - min_opacity) * curved_t
                total_shift = (i + 1) * shift_per_step
                # Use slicing for shift
                if total_shift < canvas_np.shape[1]:
                    shifted_stamp = np.zeros_like(stamp_source_np)
                    shifted_stamp[:, total_shift:] = stamp_source_np[:, :-total_shift]
                    canvas_np = np.maximum(canvas_np, shifted_stamp * current_opacity)
            processed_frames.append(torch.from_numpy(canvas_np).unsqueeze(0))
        result = torch.stack(processed_frames).to(mask.device)
        
    if added_batch:
        result = result.squeeze(0)
    return result


def apply_dubois_anaglyph_torch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Applies Dubois Red/Cyan anaglyph using Torch.

    Args:
        left: Left eye tensor [B, 3, H, W] (0.0-1.0)
        right: Right eye tensor [B, 3, H, W] (0.0-1.0)

    Returns:
        Anaglyph tensor [B, 3, H, W]
    """
    # Dubois Red-Cyan matrices
    # Left contributions
    l_mat = torch.tensor([
        [ 0.456,  0.500,  0.176], # Red
        [-0.040, -0.038, -0.016], # Green
        [-0.015, -0.021, -0.005]  # Blue
    ], device=left.device, dtype=left.dtype)

    # Right contributions
    r_mat = torch.tensor([
        [-0.043, -0.088, -0.002], # Red
        [ 0.378,  0.734, -0.018], # Green
        [-0.072, -0.113,  1.226]  # Blue
    ], device=right.device, dtype=right.dtype)

    # Reshape for matrix multiplication: [B, 3, H, W] -> [B, H*W, 3]
    B, C, H, W = left.shape
    l_flat = left.permute(0, 2, 3, 1).reshape(-1, 3)
    r_flat = right.permute(0, 2, 3, 1).reshape(-1, 3)

    # Apply matrices
    res_flat = torch.matmul(l_flat, l_mat.t()) + torch.matmul(r_flat, r_mat.t())
    
    # Reshape back and clamp
    res = res_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)
    return torch.clamp(res, 0.0, 1.0)


def apply_optimized_anaglyph_torch(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    """Applies Optimized Half-Color anaglyph using Torch.

    Args:
        left: Left eye tensor [B, 3, H, W] (0.0-1.0)
        right: Right eye tensor [B, 3, H, W] (0.0-1.0)

    Returns:
        Anaglyph tensor [B, 3, H, W]
    """
    # Red channel from left (Green/Blue weighted mix for luminance)
    # Optimized weight for Red channel: 0.0*R + 0.7*G + 0.3*B from left
    l_red = left[:, 1:2, :, :] * 0.7 + left[:, 2:3, :, :] * 0.3
    
    # Green and Blue channels from right
    res = torch.cat([l_red, right[:, 1:2, :, :], right[:, 2:3, :, :]], dim=1)
    return torch.clamp(res, 0.0, 1.0)


def apply_borders_to_frames(
    left_border_pct: float, right_border_pct: float, original_left: torch.Tensor, blended_right: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply borders to the left and right eye frames by zeroing out border pixels.
    """
    if left_border_pct <= 0 and right_border_pct <= 0:
        return original_left, blended_right

    _, _, H, W = original_left.shape

    if left_border_pct > 100 or right_border_pct > 100:
        left_px = int(round(left_border_pct))
        right_px = int(round(right_border_pct))
    else:
        left_px = int(round(W * left_border_pct / 100.0))
        right_px = int(round(W * right_border_pct / 100.0))

    if left_px < 0: left_px = 0
    if right_px < 0: right_px = 0
    if left_px >= W or right_px >= W:
        return original_left, blended_right

    left_with_border = original_left.clone()
    right_with_border = blended_right.clone()

    if left_px > 0:
        left_with_border[:, :, :, :left_px] = 0.0
    if right_px > 0:
        right_with_border[:, :, :, -right_px:] = 0.0

    return left_with_border, right_with_border


def apply_color_transfer(source_frame: torch.Tensor, target_frame: torch.Tensor) -> torch.Tensor:
    """
    Transfers the color statistics from the source_frame to the target_frame using LAB color space.
    Expects tensors in [C, H, W] or [B, C, H, W] float [0, 1] format.
    Returns 3D [C, H, W] float32 tensor.
    """
    try:
        source_for_permute = source_frame.squeeze(0) if source_frame.dim() == 4 else source_frame
        target_for_permute = target_frame.squeeze(0) if target_frame.dim() == 4 else target_frame

        source_np = source_for_permute.permute(1, 2, 0).cpu().numpy()
        target_np = target_for_permute.permute(1, 2, 0).cpu().numpy()

        source_np_uint8 = (np.clip(source_np, 0.0, 1.0) * 255).astype(np.uint8)
        target_np_uint8 = (np.clip(target_np, 0.0, 1.0) * 255).astype(np.uint8)

        source_lab = cv2.cvtColor(source_np_uint8, cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor(target_np_uint8, cv2.COLOR_RGB2LAB)

        src_mean, src_std = cv2.meanStdDev(source_lab)
        tgt_mean, tgt_std = cv2.meanStdDev(target_lab)

        src_mean = src_mean.flatten()
        src_std = src_std.flatten()
        tgt_mean = tgt_mean.flatten()
        tgt_std = tgt_std.flatten()

        src_std = np.clip(src_std, 1e-6, None)
        tgt_std = np.clip(tgt_std, 1e-6, None)

        target_lab_float = target_lab.astype(np.float32)
        for i in range(3):
            target_lab_float[:, :, i] = (target_lab_float[:, :, i] - tgt_mean[i]) / tgt_std[i] * src_std[i] + src_mean[i]

        adjusted_lab_uint8 = np.clip(target_lab_float, 0, 255).astype(np.uint8)
        adjusted_rgb = cv2.cvtColor(adjusted_lab_uint8, cv2.COLOR_LAB2RGB)
        return torch.from_numpy(adjusted_rgb).permute(2, 0, 1).float() / 255.0
    except Exception as e:
        logger.error(f"Error during color transfer: {e}")
        return target_frame


def apply_dubois_anaglyph(left_rgb_np: np.ndarray, right_rgb_np: np.ndarray) -> np.ndarray:
    """Dubois anaglyph (legacy NumPy version)."""
    left_float = left_rgb_np.astype(np.float32) / 255.0
    right_float = right_rgb_np.astype(np.float32) / 255.0
    left_matrix = np.array([[0.456, 0.500, 0.176], [-0.040, -0.038, -0.016], [-0.015, -0.021, -0.005]], dtype=np.float32)
    right_matrix = np.array([[-0.043, -0.088, -0.002], [0.378, 0.734, -0.018], [-0.072, -0.113, 1.226]], dtype=np.float32)
    H, W = left_float.shape[:2]
    left_flat = left_float.reshape(-1, 3)
    right_flat = right_float.reshape(-1, 3)
    anaglyph_flat = np.clip(np.dot(left_flat, left_matrix.T) + np.dot(right_flat, right_matrix.T), 0.0, 1.0)
    return (anaglyph_flat.reshape(H, W, 3) * 255.0).astype(np.uint8)


def apply_optimized_anaglyph(left_rgb_np: np.ndarray, right_rgb_np: np.ndarray) -> np.ndarray:
    """Optimized anaglyph (legacy NumPy version)."""
    left_float = left_rgb_np.astype(np.float32) / 255.0
    right_float = right_rgb_np.astype(np.float32) / 255.0
    left_matrix = np.array([[0.0, 0.7, 0.3], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    right_matrix = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    H, W = left_float.shape[:2]
    left_flat = left_float.reshape(-1, 3)
    right_flat = right_float.reshape(-1, 3)
    anaglyph_flat = np.clip(np.dot(left_flat, left_matrix.T) + np.dot(right_flat, right_matrix.T), 0.0, 1.0)
    return (anaglyph_flat.reshape(H, W, 3) * 255.0).astype(np.uint8)


def custom_dilate(tensor: torch.Tensor, kernel_size_x: float, kernel_size_y: float, use_gpu: bool = False, max_content_value: float = 1.0) -> torch.Tensor:
    """Applies 16-bit fractional dilation/erosion."""
    kx_raw, ky_raw = float(kernel_size_x), float(kernel_size_y)
    if abs(kx_raw) <= 1e-5 and abs(ky_raw) <= 1e-5: return tensor
    if (kx_raw > 0 and ky_raw < 0) or (kx_raw < 0 and ky_raw > 0):
        return custom_dilate(custom_dilate(tensor, kx_raw, 0, use_gpu, max_content_value), 0, ky_raw, use_gpu, max_content_value)
    is_erosion = kx_raw < 0 or ky_raw < 0
    kx_abs, ky_abs = abs(kx_raw), abs(ky_raw)
    def get_params(v):
        if v <= 1e-5: return 1, 1, 0.0
        if v < 3.0: return 1, 3, (v / 3.0)
        base = 3 + 2 * int((v - 3) // 2)
        return base, base + 2, (v - base) / 2.0
    kx_low, kx_high, tx = get_params(kx_abs)
    ky_low, ky_high, ty = get_params(ky_abs)
    tensor_cpu = tensor.to("cpu")
    processed = []
    for t in range(tensor_cpu.shape[0]):
        frame = tensor_cpu[t].numpy()
        frame_2d = frame[0] if frame.shape[0] == 1 else np.transpose(frame, (1, 2, 0))
        eff_max = max(max_content_value, 1e-5)
        src_img = np.ascontiguousarray(np.clip((frame_2d / eff_max) * 65535, 0, 65535).astype(np.uint16))
        def do_op(kw, kh, img):
            if kw <= 1 and kh <= 1: return img.astype(np.float32)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
            op = cv2.erode if is_erosion else cv2.dilate
            return op(img, kernel, iterations=1).astype(np.float32)
        if tx <= 1e-4 and ty <= 1e-4: res = do_op(kx_low, ky_low, src_img)
        elif tx > 1e-4 and ty <= 1e-4: res = (1.0 - tx) * do_op(kx_low, ky_low, src_img) + tx * do_op(kx_high, ky_low, src_img)
        elif tx <= 1e-4 and ty > 1e-4: res = (1.0 - ty) * do_op(kx_low, ky_low, src_img) + ty * do_op(kx_low, ky_high, src_img)
        else:
            r11, r12 = do_op(kx_low, ky_low, src_img), do_op(kx_low, ky_high, src_img)
            r21, r22 = do_op(kx_high, ky_low, src_img), do_op(kx_high, ky_high, src_img)
            res = (1.0 - tx) * ((1.0 - ty) * r11 + ty * r12) + tx * ((1.0 - ty) * r21 + ty * r22)
        processed.append(torch.from_numpy((res / 65535.0) * eff_max).unsqueeze(0).float())
    return torch.stack(processed).to(tensor.device)


def custom_dilate_left(tensor: torch.Tensor, kernel_size: float, use_gpu: bool = False, max_content_value: float = 1.0) -> torch.Tensor:
    """Directional 16-bit fractional dilation to the LEFT."""
    k_raw = float(kernel_size)
    if abs(k_raw) <= 1e-5: return tensor
    is_erosion, k_raw = k_raw < 0, abs(k_raw)
    def get_params(v):
        if v <= 1e-5: return 1, 1, 0.0
        if v < 3.0: return 1, 3, (v / 3.0)
        base = 3 + 2 * int((v - 3) // 2)
        return base, base + 2, (v - base) / 2.0
    kw_low, kw_high, t = get_params(k_raw)
    k_low, k_high = int(kw_low // 2), int(kw_high // 2)
    if k_low <= 0 and k_high <= 0: return tensor
    eff_max = max(float(max_content_value), 1e-5)
    tensor_cpu = tensor.to("cpu")
    def do_op(k_int, src):
        if k_int <= 0: return src.astype(np.float32)
        kernel = np.ones((1, k_int + 1), dtype=np.uint8)
        op = cv2.erode if is_erosion else cv2.dilate
        return op(src, kernel, anchor=(0, 0), iterations=1).astype(np.float32)
    processed = []
    for idx in range(tensor_cpu.shape[0]):
        f = tensor_cpu[idx].numpy()
        f2d = f[0] if f.shape[0] == 1 else np.transpose(f, (1, 2, 0))
        src = np.ascontiguousarray(np.clip((f2d / eff_max) * 65535, 0, 65535).astype(np.uint16)).astype(np.float32)
        out = do_op(k_low, src) if abs(t) <= 1e-4 else (1.0 - t) * do_op(k_low, src) + t * do_op(k_high, src)
        out_f = (np.clip(out, 0, 65535).astype(np.float32) / 65535.0) * eff_max
        processed.append(out_f[None, ...] if f.shape[0] == 1 else np.transpose(out_f, (2, 0, 1)))
    return torch.from_numpy(np.stack(processed, axis=0)).to(tensor.device)


def custom_blur_left_masked(tensor_before: torch.Tensor, tensor_after: torch.Tensor, kernel_size: int, use_gpu: bool = False, max_content_value: float = 1.0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Applies Gaussian blur ONLY to pixels that changed."""
    k = int(float(kernel_size))
    if k <= 0: return tensor_after
    if k % 2 == 0: k += 1
    k = max(k, 1)
    if mask is None:
        changed = (tensor_after - tensor_before).abs() > 1e-12
    else:
        changed = mask.unsqueeze(1) if mask.ndim == 3 else mask
        changed = (changed > 0.5) if changed.dtype != torch.bool else changed
    if not bool(changed.any().item()): return tensor_after
    return torch.where(changed.to(tensor_after.device), custom_blur(tensor_after, k, k, use_gpu, max_content_value), tensor_after)


def custom_blur(tensor: torch.Tensor, kernel_size_x: int, kernel_size_y: int, use_gpu: bool = True, max_content_value: float = 1.0) -> torch.Tensor:
    """Applies 16-bit Gaussian blur."""
    kx, ky = int(kernel_size_x), int(kernel_size_y)
    if kx <= 0 and ky <= 0: return tensor
    kx, ky = (kx if kx % 2 == 1 else kx + 1), (ky if ky % 2 == 1 else ky + 1)
    tensor_cpu = tensor.to("cpu")
    processed = []
    for t in range(tensor_cpu.shape[0]):
        f = tensor_cpu[t].numpy()
        f2d = f[0] if f.shape[0] == 1 else np.transpose(f, (1, 2, 0))
        eff_max = max(max_content_value, 1e-5)
        src = np.ascontiguousarray(np.clip((f2d / eff_max) * 65535, 0, 65535).astype(np.uint16))
        out = cv2.GaussianBlur(src, (kx, ky), 0)
        processed.append(torch.from_numpy((out.astype(np.float32) / 65535.0) * eff_max).unsqueeze(0).float())
    return torch.stack(processed).to(tensor.device)
