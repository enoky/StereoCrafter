# Inpainting GUI Change Log

## Version 26-02-21.0

### Added

- **Single Clip ID Filter**: Added optional clip ID filter for batch processing (e.g., `*-0006_*`). When used, processed files remain in their original folder instead of moving to 'finished'.
- **Advanced Inpainting Mask Pipeline**:
  - **Thresholding**: Added `inpaint_mask_initial_threshold` (pre-morph) and `inpaint_mask_post_threshold` (final binarization).
  - **Refinement**: Added `inpaint_mask_morph_kernel_size`, `inpaint_mask_dilate_kernel_size`, and `inpaint_mask_blur_kernel_size` to fine-tune masks before inference.
- **Inpaint Cache**: Added `keep_inpaint_cache` option to store intermediate inpainting results, allowing for faster re-runs when only changing blending parameters.
- **Hi-Res Blending Enhancements**:
  - **Mask Source Selection**: New `blend_mask_source` option to choose between 'lowres', 'hires', or 'hybrid' mask sources during hi-res blending.
  - **Soft Mask Support**: Added `mask_post_threshold` to allow for soft mask values (transparent blending) or final binarization.

### Changed

- **Blend Mask Thresholding**: Updated `mask_initial_threshold` to serve as a pre-threshold before morphological operations, improving mask consistency.

#### *These addtion where obtained from [46cv8](https://github.com/46cv8/StereoCrafter/tree/002_fix_inpaint_errors)*

---

## Version 26-01-13.0

- Initial modularized version of the Inpainting GUI.
