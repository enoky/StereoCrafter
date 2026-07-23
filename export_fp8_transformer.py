"""One-time export of an FP8 weight-only quantized copy of the V2 inpaint transformer.

Quantizes ./weights/StereoCrafter2 (Wan VACE 14B, bf16) to FP8-e4m3 weight-only
via torchao and writes ./weights/StereoCrafter2-FP8/ (config.json + quantized
state dict, ~16 GiB). The result is loaded by inpainting_gui_v2.py's
"FP8 resident" offload mode, which needs a ~24 GB+ GPU (e.g. RTX 4090/5090).

Run on a machine with ~64 GB system RAM (peak RSS during quantization is ~50 GB):

    uv run python export_fp8_transformer.py

The output folder can then be copied to other machines; loading it needs only
~17 GB RAM because the bf16 model is never materialized.
"""

import os
import time

import torch
from accelerate import init_empty_weights
from diffusers import WanVACETransformer3DModel
from torchao.quantization import quantize_, Float8WeightOnlyConfig

SRC = "./weights/StereoCrafter2"
DST = "./weights/StereoCrafter2-FP8"
STATE_FILE = "diffusion_pytorch_model_fp8.pt"


def main():
    t0 = time.perf_counter()
    print(f"Loading bf16 transformer from {SRC} ...", flush=True)
    tf = WanVACETransformer3DModel.from_pretrained(SRC, torch_dtype=torch.bfloat16)
    tf.eval().requires_grad_(False)
    print(f"  loaded in {time.perf_counter() - t0:.0f}s", flush=True)

    t0 = time.perf_counter()
    print("Quantizing to FP8 weight-only (CPU)...", flush=True)
    quantize_(tf, Float8WeightOnlyConfig())
    print(f"  quantized in {time.perf_counter() - t0:.0f}s", flush=True)

    os.makedirs(DST, exist_ok=True)
    tf.save_config(DST)
    t0 = time.perf_counter()
    print(f"Saving quantized state dict to {DST}/{STATE_FILE} ...", flush=True)
    torch.save(tf.state_dict(), os.path.join(DST, STATE_FILE))
    size = os.path.getsize(os.path.join(DST, STATE_FILE)) / 2**30
    print(f"  saved {size:.1f} GiB in {time.perf_counter() - t0:.0f}s", flush=True)

    # ---------------- verification: reload the way the GUI will ----------------
    print("Verifying roundtrip (meta-init + assign load)...", flush=True)
    cfg = WanVACETransformer3DModel.load_config(DST)
    with init_empty_weights():
        tf2 = WanVACETransformer3DModel.from_config(cfg)
    sd = torch.load(os.path.join(DST, STATE_FILE), map_location="cpu", weights_only=False)
    missing, unexpected = tf2.load_state_dict(sd, assign=True, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    tf2.eval().requires_grad_(False)

    # no tensor (param, buffer, or plain attribute) may remain on meta
    meta = [n for n, p in tf2.named_parameters() if p.device.type == "meta"]
    meta += [n for n, b in tf2.named_buffers() if b.device.type == "meta"]
    assert not meta, f"tensors left on meta: {meta[:5]}"

    # sample-tensor fidelity vs the in-memory quantized model
    sd_ref = tf.state_dict()
    keys = list(sd_ref.keys())[:: max(1, len(sd_ref) // 50)]
    for k in keys:
        a, b = sd_ref[k], sd[k]
        assert type(a) is type(b), (k, type(a), type(b))
        fa = a.dequantize() if hasattr(a, "dequantize") else a
        fb = b.dequantize() if hasattr(b, "dequantize") else b
        assert torch.equal(fa, fb), f"mismatch at {k}"
    print(f"  {len(keys)} sampled tensors bit-identical after roundtrip", flush=True)

    # tiny full-graph CPU forward to prove the reloaded model is runnable
    # (catches meta-device leftovers in non-buffer attributes, e.g. rope tables)
    h = torch.randn(1, 16, 1, 8, 8, dtype=torch.bfloat16)
    ctrl = torch.randn(1, 96, 1, 8, 8, dtype=torch.bfloat16)
    scale = torch.ones(len(tf2.config.vace_layers), dtype=torch.bfloat16)
    text = torch.randn(1, 16, 4096, dtype=torch.bfloat16)
    ts = torch.tensor([500])
    t0 = time.perf_counter()
    with torch.no_grad():
        out = tf2(hidden_states=h, timestep=ts, encoder_hidden_states=text,
                  control_hidden_states=ctrl, control_hidden_states_scale=scale,
                  return_dict=False)[0]
    assert torch.isfinite(out).all()
    print(f"  tiny CPU forward OK in {time.perf_counter() - t0:.0f}s "
          f"(out std {out.float().std():.4f})", flush=True)
    print("EXPORT COMPLETE:", DST, flush=True)


if __name__ == "__main__":
    main()
