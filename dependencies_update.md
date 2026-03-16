# Dependencies Update Log

Tracks modifications made to files outside the designated WEBUI folders (`webui.py`, `gui/`, `stereocrafter_ui/`).

---

## `core/common/video_io.py`

### Change 1: Reorder imports — torch before decord

**What changed:**
```python
# Before
from decord import VideoReader, cpu

# After
import torch                          # imported first
from core.common.gpu_utils import CUDA_AVAILABLE
from core.common.encoding_utils import build_encoder_args, get_encoding_config_from_dict
from decord import VideoReader, cpu   # imported after torch
```

**Why:**
On Windows, importing `decord` before `torch` causes a DLL initialization conflict (`WinError 1114`). Importing `torch` first ensures CUDA DLLs are loaded in the correct order before decord tries to access them.

Reference: https://github.com/dmlc/decord/issues/174

---

### Change 2: Fix `encoding_options` NameError in `encode_frames_to_mp4`

**What changed:**
```python
# Before
enc_config = get_encoding_config_from_dict(encoding_options or {})

# After
enc_config = get_encoding_config_from_dict({})  # Use empty dict as default
```

**Why:**
The variable `encoding_options` was referenced but never defined in the function scope, causing a `NameError` at runtime when called from the inpainting pipeline. The original GUI avoided this by always passing the argument explicitly. The WebUI call path did not, so the fallback was hardcoded to an empty dict (which triggers the same default encoding behaviour).

---
