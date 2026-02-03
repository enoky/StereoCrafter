"""Preview rendering module for various display modes.

Handles preview frame generation for various display modes including:
- Splat Result (Full/Low resolution)
- Occlusion Mask
- Depth Map (raw and colorized)
- Anaglyph 3D (Dubois and Optimized)
- Wigglegram

This module provides the rendering logic extracted from the main GUI
to enable preview generation for the splatting workflow.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from dependency.stereocrafter_util import (
    apply_dubois_anaglyph,
    apply_optimized_anaglyph,
    release_cuda_memory,
)

from .forward_warp import ForwardWarpStereo

logger = logging.getLogger(__name__)

# Constants
DEPTH_VIS_APPLY_TV_RANGE_EXPANSION_10BIT = True


class PreviewRenderer:
    """Handles preview frame generation for various display modes.

    Supports multiple preview modes including:
    - Splat Result (Full and Low resolution variants)
    - Occlusion Mask
    - Depth Map (raw grayscale and colorized)
    - Anaglyph 3D (standard, Dubois, Optimized)
    - Wigglegram
    - Original (Left Eye)
    """

    # Supported preview modes
    MODES = {
        "splat": ["Splat Result", "Splat Result(Low)"],
        "occlusion": ["Occlusion Mask", "Occlusion Mask(Low)"],
        "depth_raw": ["Depth Map"],
        "depth_color": ["Depth Map (Color)"],
        "original": ["Original (Left Eye)"],
        "anaglyph": ["Anaglyph 3D"],
        "anaglyph_dubois": ["Dubois Anaglyph"],
        "anaglyph_optimized": ["Optimized Anaglyph"],
        "wigglegram": ["Wigglegram"],
    }

    def __init__(self, cuda_available: bool = True):
        """Initialize preview renderer.

        Args:
            cuda_available: Whether CUDA is available for GPU processing
        """
        self.logger = logging.getLogger(__name__)
        self.cuda_available = cuda_available

    def find_preview_sources(
        self,
        source_path: str,
        depth_path: str,
        multi_map: bool = False,
    ) -> List[Dict[str, str]]:
        """Find matching source video and depth map pairs for preview.

        Scans for matching source video and depth map pairs, handling both
        single-file mode and folder/batch modes. Supports Multi-Map mode
        where depth maps are organized in subfolders.

        Args:
            source_path: Path to source video file or folder
            depth_path: Path to depth map file or folder
            multi_map: Whether Multi-Map mode is enabled

        Returns:
            List of dictionaries with 'source_video' and 'depth_map' keys
        """
        import glob

        if not source_path or not depth_path:
            self.logger.warning("Preview Scan Failed: Source or depth path is empty.")
            return []

        # Single-file mode
        is_source_file = os.path.isfile(source_path)
        is_depth_file = os.path.isfile(depth_path)

        if is_source_file and is_depth_file:
            self.logger.debug(
                f"Preview Scan: Single file mode. Source: {source_path}, Depth: {depth_path}"
            )
            return [{"source_video": source_path, "depth_map": depth_path}]

        # Folder/batch mode
        if not os.path.isdir(source_path) or not os.path.isdir(depth_path):
            self.logger.error(
                "Preview Scan Failed: Inputs must be two files or two valid directories."
            )
            return []

        video_extensions = ("*.mp4", "*.avi", "*.mov", "*.mkv")
        source_videos = []
        for ext in video_extensions:
            source_videos.extend(glob.glob(os.path.join(source_path, ext)))

        if not source_videos:
            self.logger.warning(f"No source videos found in folder: {source_path}")
            return []

        video_source_list = []

        if multi_map:
            # Multi-Map mode: search all map subfolders
            depth_candidate_folders = []
            try:
                for entry in os.listdir(depth_path):
                    full_sub = os.path.join(depth_path, entry)
                    if os.path.isdir(full_sub) and entry.lower() != "sidecars":
                        depth_candidate_folders.append(full_sub)
            except FileNotFoundError:
                self.logger.error(
                    f"Preview Scan Failed: Depth folder not found: {depth_path}"
                )
                return []

            for video_path in sorted(source_videos):
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                matched = False

                for dpath in depth_candidate_folders:
                    mp4 = os.path.join(dpath, f"{base_name}_depth.mp4")
                    npz = os.path.join(dpath, f"{base_name}_depth.npz")

                    if os.path.exists(mp4):
                        video_source_list.append(
                            {"source_video": video_path, "depth_map": mp4}
                        )
                        matched = True
                        break
                    elif os.path.exists(npz):
                        video_source_list.append(
                            {"source_video": video_path, "depth_map": npz}
                        )
                        matched = True
                        break

                if not matched:
                    self.logger.debug(
                        f"Preview Scan: No depth map found for '{base_name}'."
                    )
        else:
            # Normal mode: single depth folder
            for video_path in sorted(source_videos):
                base_name = os.path.splitext(os.path.basename(video_path))[0]

                candidates = [
                    os.path.join(depth_path, f"{base_name}_depth.mp4"),
                    os.path.join(depth_path, f"{base_name}_depth.npz"),
                    os.path.join(depth_path, f"{base_name}.mp4"),
                    os.path.join(depth_path, f"{base_name}.npz"),
                ]

                matching_depth_path = None
                for dp in candidates:
                    if os.path.exists(dp):
                        matching_depth_path = dp
                        break

                if matching_depth_path:
                    video_source_list.append(
                        {
                            "source_video": video_path,
                            "depth_map": matching_depth_path,
                        }
                    )

        if not video_source_list:
            self.logger.warning("Preview Scan: No matching source/depth pairs found.")
        else:
            self.logger.info(
                f"Preview Scan: Found {len(video_source_list)} matching pairs."
            )

        return video_source_list

    def render_preview_frame(
        self,
        source_frame: torch.Tensor,
        depth_frame: torch.Tensor,
        settings: Dict,
        preview_mode: str,
        wigglegram_callback=None,
    ) -> Optional[Image.Image]:
        """Render a preview frame based on the selected mode.

        Performs splatting and renders the result in the requested preview mode.
        Supports various output modes including anaglyph 3D, depth visualization,
        and wigglegram animation.

        Args:
            source_frame: Source video frame tensor [1, 3, H, W]
            depth_frame: Depth map frame tensor [1, 1, H, W] or [1, C, H, W]
            settings: Dictionary of processing parameters
            preview_mode: Preview mode string (e.g., 'Splat Result', 'Anaglyph 3D')
            wigglegram_callback: Optional callback for wigglegram animation

        Returns:
            Rendered PIL Image, or None for Wigglegram mode (handled by callback)
        """
        if not self.cuda_available:
            self.logger.error("Preview processing requires a CUDA-enabled GPU.")
            return None

        self.logger.debug("--- Starting Preview Render ---")

        # Check inputs
        if source_frame is None or depth_frame is None:
            self.logger.error("Preview failed: Missing source or depth frame.")
            return None

        # Get settings
        if not settings:
            self.logger.error("Preview failed: No settings provided.")
            return None

        # Determine if low-res preview
        is_low_res = preview_mode in ["Splat Result(Low)", "Occlusion Mask(Low)"]

        # Get original dimensions
        W_orig = source_frame.shape[3]
        H_orig = source_frame.shape[2]

        # Setup target resolution
        W_target, H_target = W_orig, H_orig

        # Handle low-res preview sizing
        if is_low_res:
            W_target, H_target = self._calculate_low_res_dimensions(
                W_orig, H_orig, settings.get("target_width", 0)
            )

            try:
                source_resized = F.interpolate(
                    source_frame.cuda(),
                    size=(H_target, W_target),
                    mode="bilinear",
                    align_corners=False,
                )
            except Exception as e:
                self.logger.error(
                    f"Low-Res preview resize failed: {e}. Falling back to original."
                )
                W_target, H_target = W_orig, H_orig
                source_resized = source_frame.cuda()
        else:
            source_resized = source_frame.cuda()

        # Process depth frame
        depth_processed = self._process_depth_for_preview(
            depth_frame, W_orig, H_orig, settings, is_low_res, W_target, H_target
        )

        if depth_processed is None:
            return None

        # Perform splatting
        stereo_projector = ForwardWarpStereo(occlu_map=True).cuda()

        # Resize depth to target resolution
        disp_map = (
            torch.from_numpy(depth_processed)
            .unsqueeze(0)
            .unsqueeze(0)
            .float()
            .cuda()
        )

        if H_target != disp_map.shape[2] or W_target != disp_map.shape[3]:
            disp_map = F.interpolate(
                disp_map,
                size=(H_target, W_target),
                mode="bilinear",
                align_corners=False,
            )

        # Calculate disparity
        convergence = float(settings.get("convergence_point", 0.5))
        max_disp = float(settings.get("max_disp", 20.0))
        tv_comp = float(settings.get("tv_disp_compensation", 1.0))

        disp_map = (disp_map - convergence) * 2.0
        actual_max_disp_pixels = (max_disp / 20.0 / 100.0) * W_target * tv_comp
        disp_map = disp_map * actual_max_disp_pixels

        # Perform forward warp
        with torch.no_grad():
            right_eye_raw, occlusion_mask = stereo_projector(source_resized, disp_map)
            right_eye = right_eye_raw

        # Apply borders for anaglyph and wigglegram
        left_pct = settings.get("left_border_pct", 0.0)
        right_pct = settings.get("right_border_pct", 0.0)

        if preview_mode in [
            "Anaglyph 3D",
            "Dubois Anaglyph",
            "Optimized Anaglyph",
            "Wigglegram",
        ]:
            l_px = int(round(left_pct * W_target / 100.0))
            r_px = int(round(right_pct * W_target / 100.0))

            if l_px > 0:
                source_resized[:, :, :, :l_px] = 0.0
            if r_px > 0:
                right_eye[:, :, :, -r_px:] = 0.0

        # Render based on mode
        final_tensor = self._render_by_mode(
            source_resized,
            right_eye,
            depth_processed,
            occlusion_mask,
            preview_mode,
        )

        # Handle wigglegram special case
        if preview_mode == "Wigglegram" and wigglegram_callback:
            wigglegram_callback(source_resized.cpu(), right_eye.cpu())
            del stereo_projector, disp_map, right_eye_raw, occlusion_mask
            release_cuda_memory()
            return None

        # Convert to PIL Image
        if final_tensor is not None:
            pil_img = Image.fromarray(
                (final_tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
        else:
            pil_img = None

        # Cleanup
        del stereo_projector, disp_map, right_eye_raw, occlusion_mask
        release_cuda_memory()

        self.logger.debug("--- Finished Preview Render ---")
        return pil_img

    def _calculate_low_res_dimensions(
        self, W_orig: int, H_orig: int, target_width: int
    ) -> Tuple[int, int]:
        """Calculate aspect-ratio-correct low-res dimensions.

        Args:
            W_orig: Original width
            H_orig: Original height
            target_width: Requested target width

        Returns:
            Tuple of (W_target, H_target) divisible by 2
        """
        if target_width <= 0:
            return W_orig, H_orig

        aspect_ratio = W_orig / H_orig
        H_calculated = int(round(target_width / aspect_ratio))

        # Ensure divisible by 2
        W_target = target_width if target_width % 2 == 0 else target_width + 1
        H_target = H_calculated if H_calculated % 2 == 0 else H_calculated + 1

        if W_target <= 0 or H_target <= 0:
            return W_orig, H_orig

        return W_target, H_target

    def _process_depth_for_preview(
        self,
        depth_frame: torch.Tensor,
        W_orig: int,
        H_orig: int,
        settings: Dict,
        is_low_res: bool,
        W_target: int,
        H_target: int,
    ) -> Optional[np.ndarray]:
        """Process depth frame for preview rendering.

        Handles depth preprocessing including normalization, gamma correction,
        and optional resizing for low-res previews.

        Args:
            depth_frame: Raw depth frame tensor
            W_orig: Original width
            H_orig: Original height
            settings: Processing settings dictionary
            is_low_res: Whether this is a low-res preview
            W_target: Target width
            H_target: Target height

        Returns:
            Processed depth as numpy array [H, W], or None on error
        """
        # Convert to numpy
        depth_numpy = depth_frame.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Ensure 3D
        if depth_numpy.ndim == 2:
            depth_numpy = depth_numpy[:, :, None]

        # Resize to original resolution if needed
        if depth_numpy.shape[0] != H_orig or depth_numpy.shape[1] != W_orig:
            try:
                interp = (
                    cv2.INTER_LINEAR
                    if (W_orig > depth_numpy.shape[1] or H_orig > depth_numpy.shape[0])
                    else cv2.INTER_AREA
                )
                depth_numpy = cv2.resize(depth_numpy, (W_orig, H_orig), interpolation=interp)
                if depth_numpy.ndim == 2:
                    depth_numpy = depth_numpy[:, :, None]
            except Exception as e:
                self.logger.error(f"Depth resize failed: {e}")

        # Get max raw value for scaling
        max_raw = depth_numpy.max()
        if max_raw < 1.0:
            max_raw = 1.0

        # Determine scaling factor
        if max_raw <= 256.0 and max_raw > 1.0:
            scale = 255.0
        elif max_raw > 256.0 and max_raw <= 1024.0:
            scale = max_raw
        elif max_raw > 1024.0:
            scale = 65535.0
        else:
            scale = 1.0

        # Normalize to 0-1
        depth_normalized = depth_numpy.squeeze() / scale
        depth_normalized = np.clip(depth_normalized, 0, 1)

        # Apply gamma
        gamma = float(settings.get("depth_gamma", 1.0))
        if round(gamma, 2) != 1.0:
            depth_normalized = 1.0 - np.power(
                1.0 - depth_normalized, gamma
            )
            depth_normalized = np.clip(depth_normalized, 0, 1)

        # Low-res: resize processed depth
        if is_low_res and (depth_normalized.shape[0] != H_target or depth_normalized.shape[1] != W_target):
            try:
                interp = (
                    cv2.INTER_AREA
                    if (W_target < depth_normalized.shape[1] and H_target < depth_normalized.shape[0])
                    else cv2.INTER_LINEAR
                )
                depth_normalized = cv2.resize(depth_normalized, (W_target, H_target), interpolation=interp)
            except Exception as e:
                self.logger.error(f"Low-res depth resize failed: {e}")

        return depth_normalized

    def _render_by_mode(
        self,
        left_eye: torch.Tensor,
        right_eye: torch.Tensor,
        depth: np.ndarray,
        occlusion: torch.Tensor,
        mode: str,
    ) -> Optional[torch.Tensor]:
        """Render output based on preview mode.

        Args:
            left_eye: Left eye tensor [1, 3, H, W]
            right_eye: Right eye tensor [1, 3, H, W]
            depth: Normalized depth array [H, W]
            occlusion: Occlusion mask tensor [1, 1, H, W]
            mode: Preview mode string

        Returns:
            Final output tensor [1, 3, H, W], or None
        """
        if mode in ["Splat Result", "Splat Result(Low)"]:
            return right_eye.cpu()

        elif mode in ["Occlusion Mask", "Occlusion Mask(Low)"]:
            return occlusion.repeat(1, 3, 1, 1).cpu()

        elif mode == "Depth Map":
            return self._render_depth_raw(depth)

        elif mode == "Depth Map (Color)":
            return self._render_depth_color(depth)

        elif mode == "Original (Left Eye)":
            return left_eye.cpu()

        elif mode == "Anaglyph 3D":
            return self._render_anaglyph_simple(left_eye, right_eye)

        elif mode == "Dubois Anaglyph":
            return self._render_anaglyph_dubois(left_eye, right_eye)

        elif mode == "Optimized Anaglyph":
            return self._render_anaglyph_optimized(left_eye, right_eye)

        else:
            # Default to splat result
            return right_eye.cpu()

    def _render_depth_raw(self, depth: np.ndarray) -> torch.Tensor:
        """Render raw grayscale depth map.

        Args:
            depth: Normalized depth array [H, W]

        Returns:
            RGB tensor [1, 3, H, W]
        """
        depth_uint8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        depth_3ch = np.stack([depth_uint8] * 3, axis=-1)
        return (
            torch.from_numpy(depth_3ch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )

    def _render_depth_color(self, depth: np.ndarray) -> torch.Tensor:
        """Render colorized depth map.

        Args:
            depth: Normalized depth array [H, W]

        Returns:
            RGB tensor [1, 3, H, W]
        """
        depth_uint8 = (np.clip(depth, 0, 1) * 255).astype(np.uint8)
        vis_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
        vis_rgb = cv2.cvtColor(vis_color, cv2.COLOR_BGR2RGB)
        return (
            torch.from_numpy(vis_rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        )

    def _render_anaglyph_simple(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """Render simple red-cyan anaglyph.

        Args:
            left: Left eye tensor [1, 3, H, W]
            right: Right eye tensor [1, 3, H, W]

        Returns:
            Anaglyph tensor [1, 3, H, W]
        """
        left_np = (left.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )
        right_np = (right.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )

        left_gray = cv2.cvtColor(left_np, cv2.COLOR_RGB2GRAY)
        anaglyph = right_np.copy()
        anaglyph[:, :, 0] = left_gray

        return torch.from_numpy(anaglyph).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def _render_anaglyph_dubois(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """Render Dubois anaglyph.

        Args:
            left: Left eye tensor [1, 3, H, W]
            right: Right eye tensor [1, 3, H, W]

        Returns:
            Anaglyph tensor [1, 3, H, W]
        """
        left_np = (left.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )
        right_np = (right.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )

        anaglyph = apply_dubois_anaglyph(left_np, right_np)

        return torch.from_numpy(anaglyph).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    def _render_anaglyph_optimized(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        """Render Optimized anaglyph.

        Args:
            left: Left eye tensor [1, 3, H, W]
            right: Right eye tensor [1, 3, H, W]

        Returns:
            Anaglyph tensor [1, 3, H, W]
        """
        left_np = (left.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )
        right_np = (right.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(
            np.uint8
        )

        anaglyph = apply_optimized_anaglyph(left_np, right_np)

        return torch.from_numpy(anaglyph).permute(2, 0, 1).float().unsqueeze(0) / 255.0
