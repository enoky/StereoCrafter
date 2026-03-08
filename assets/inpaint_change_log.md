# Inpainting GUI Change Log

## Version 26-03-08.0

### Added (v26-03-08.0)

- **Unified Encoding Settings Dialog**: Added encoding options accessible via Options → Encoding Settings....
  - Previously only CRF was available in main UI, now moved to unified dialog
  - **Encoder**: Auto (NVENC/CPU) or Force CPU
  - **Quality Preset**: Fastest to Slowest (controls encoding speed/efficiency)
  - **CPU Tune**: Film, Grain, Animation, etc. (ignored for NVENC)
  - **CRF**: Constant Rate Factor for video quality control
  - **NVENC Options**: Lookahead, Spatial AQ, Temporal AQ, AQ Strength
  - **Color Tags**: Off, Auto, BT.709 L/F, BT.2020 PQ/HLG
- Implemented as reusable `core/ui/encoding_settings.py` shared across all three GUIs
- Removed CRF field from main parameter UI (now in encoding dialog)

## Version 26-02-22.0

- **Temporal Cross-Fading**: Implemented linear temporal blending between segment chunks. Transitions are now smooth instead of abrupt "hard cuts" at overlap boundaries.
- **Improved Chunk Continuity**: The transition blend is included in mid-inference checkpoints, ensuring continuity is preserved even after a stop/resume.

---

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
