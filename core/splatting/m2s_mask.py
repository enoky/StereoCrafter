import torch


def build_m2s_occlusion_mask_from_disparity(
    disp_map_tensor: torch.Tensor,
    output_dtype=None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Build an M2S-style hole mask from StereoCrafter's final disparity tensor.

    StereoCrafter's RGB warp stays untouched. We only derive occupancy from the
    same final disparity map SC already uses for splatting, with the same
    effective horizontal destination:

        destination_x = x - disparity

    Returns (B, 1, H, W) with 1.0 for holes and 0.0 for filled pixels.
    """
    if disp_map_tensor is None:
        raise ValueError('disp_map_tensor is required')
    if disp_map_tensor.ndim != 4 or int(disp_map_tensor.shape[1]) != 1:
        raise ValueError(
            f'disp_map_tensor must have shape (B, 1, H, W); got {tuple(disp_map_tensor.shape)}'
        )

    if output_dtype is None:
        output_dtype = disp_map_tensor.dtype

    disp = disp_map_tensor[:, 0].to(dtype=torch.float32)
    batch, height, width = disp.shape
    device = disp.device
    base_x = torch.arange(width, device=device, dtype=torch.float32).view(1, width)

    hole_masks = []
    for batch_index in range(batch):
        disp_frame = disp[batch_index]
        dest_x = base_x - disp_frame

        x0 = torch.floor(dest_x)
        x1 = x0 + 1.0
        w1 = dest_x - x0
        w0 = 1.0 - w1

        valid0 = (x0 >= 0.0) & (x0 < float(width))
        valid1 = (x1 >= 0.0) & (x1 < float(width))

        x0_idx = x0.clamp(0, width - 1).to(torch.int64)
        x1_idx = x1.clamp(0, width - 1).to(torch.int64)

        occupancy = torch.zeros((height, width), device=device, dtype=torch.float32)
        occupancy.scatter_add_(1, x0_idx, w0 * valid0.to(torch.float32))
        occupancy.scatter_add_(1, x1_idx, w1 * valid1.to(torch.float32))

        hole_masks.append((occupancy <= eps).to(dtype=torch.float32))

    mask = torch.stack(hole_masks, dim=0).unsqueeze(1)
    return mask.to(dtype=output_dtype).clamp_(0.0, 1.0)
