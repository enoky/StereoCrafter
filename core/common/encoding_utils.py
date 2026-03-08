import logging
from typing import Optional, Dict, Any, List
from core.common.gpu_utils import CUDA_AVAILABLE

logger = logging.getLogger(__name__)

QUALITY_PRESETS = ("Fastest", "Faster", "Fast", "Medium", "Slow", "Slower", "Slowest")

CODEC_OPTIONS = ("Auto", "H.264", "H.265")

CPU_PRESET_MAP = {
    "Fastest": "ultrafast",
    "Faster": "faster",
    "Fast": "fast",
    "Medium": "medium",
    "Slow": "slow",
    "Slower": "slower",
    "Slowest": "veryslow",
}

NVENC_PRESET_MAP = {
    "Fastest": "p1",
    "Faster": "p2",
    "Fast": "p3",
    "Medium": "p4",
    "Slow": "p5",
    "Slower": "p6",
    "Slowest": "p7",
}

CPU_TUNE_OPTIONS = ("None", "Film", "Grain", "Animation", "Still Image", "PSNR", "SSIM", "Fast Decode", "Zero Latency")

ENCODER_OPTIONS = ("Auto", "Force CPU")

DEFAULT_ENCODING_CONFIG = {
    "codec": "H.265",
    "encoder": "Auto",
    "quality": "Medium",
    "tune": "None",
    "crf": 23,
    "nvenc_lookahead_enabled": False,
    "nvenc_lookahead": 16,
    "nvenc_spatial_aq": False,
    "nvenc_temporal_aq": False,
    "nvenc_aq_strength": 8,
}


def get_encoder_codec(codec: str = "H.265", force_10bit: bool = False, encoder: str = "Auto") -> str:
    """Determine the encoder codec based on settings.

    Args:
        codec: "Auto", "H.264", or "H.265"
        force_10bit: Whether to use 10-bit encoding
        encoder: "Auto" (use NVENC if available) or "Force CPU"

    Returns:
        Codec string (e.g., "h264_nvenc", "libx264", "hevc_nvenc", "libx265")
    """
    use_nvenc = encoder != "Force CPU"

    if codec == "H.264":
        if force_10bit:
            logger.warning("H.264 does not support 10-bit encoding well. Forcing H.265.")
            codec = "H.265"
        if use_nvenc and CUDA_AVAILABLE:
            return "h264_nvenc"
        return "libx264"

    if codec == "H.265" or codec == "Auto":
        if force_10bit:
            if use_nvenc and CUDA_AVAILABLE:
                return "hevc_nvenc"
            return "libx265"
        else:
            if use_nvenc and CUDA_AVAILABLE:
                return "h264_nvenc" if codec == "Auto" else "hevc_nvenc"
            return "libx265"

    if use_nvenc and CUDA_AVAILABLE:
        return "hevc_nvenc" if force_10bit else "h264_nvenc"

    return "libx265" if force_10bit else "libx264"


def quality_to_preset(quality: str, is_nvenc: bool) -> str:
    """Convert quality preset name to FFmpeg preset.

    Args:
        quality: Quality preset name (Fastest to Slowest)
        is_nvenc: Whether using NVENC encoder

    Returns:
        FFmpeg preset string
    """
    preset_map = NVENC_PRESET_MAP if is_nvenc else CPU_PRESET_MAP
    return preset_map.get(quality, "medium" if not is_nvenc else "p4")


def get_tune_flag(tune: str, codec: str) -> Optional[str]:
    """Get the FFmpeg tune flag.

    Args:
        tune: Tune name
        codec: Codec being used

    Returns:
        FFmpeg tune flag or None
    """
    if tune == "None" or not tune:
        return None

    if "nvenc" in codec:
        logger.debug("Tune is ignored when using NVENC encoder")
        return None

    tune_map = {
        "Film": "film",
        "Grain": "grain",
        "Animation": "animation",
        "Still Image": "stillimage",
        "PSNR": "psnr",
        "SSIM": "ssim",
        "Fast Decode": "fastdecode",
        "Zero Latency": "zerolatency",
    }
    return tune_map.get(tune)


def build_encoder_args(
    codec: str = "H.265",
    encoder: str = "Auto",
    quality: str = "Medium",
    tune: str = "None",
    crf: int = 23,
    force_10bit: bool = False,
    nvenc_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build encoding arguments for FFmpeg.

    Args:
        codec: Codec selection ("Auto", "H.264", or "H.265")
        encoder: Encoder selection ("Auto" or "Force CPU")
        quality: Quality preset ("Fastest" to "Slowest")
        tune: CPU tune option
        crf: CRF value for quality control
        force_10bit: Whether to force 10-bit output
        nvenc_options: Optional dict with NVENC-specific options:
            - lookahead_enabled: bool
            - lookahead: int (frames)
            - spatial_aq: bool
            - temporal_aq: bool
            - aq_strength: int

    Returns:
        Dict with keys: codec, encoder, preset, tune, crf, pix_fmt, extra_args
    """
    codec_result = get_encoder_codec(codec, force_10bit, encoder)
    is_nvenc = "nvenc" in codec_result
    preset = quality_to_preset(quality, is_nvenc)
    tune_flag = get_tune_flag(tune, codec_result)

    pix_fmt = "yuv420p10le" if force_10bit else "yuv420p"

    extra_args = []

    if is_nvenc:
        extra_args.extend(["-qp", str(crf)])

        if nvenc_options:
            if nvenc_options.get("lookahead_enabled", False):
                la_frames = nvenc_options.get("lookahead", 16)
                extra_args.extend(["-rc-lookahead", str(la_frames)])

            if nvenc_options.get("spatial_aq", False):
                extra_args.extend(["-aq-strength", str(nvenc_options.get("aq_strength", 8))])
                extra_args.append("-spatial-aq", "1")

            if nvenc_options.get("temporal_aq", False):
                extra_args.extend(["-aq-strength", str(nvenc_options.get("aq_strength", 8))])
                extra_args.append("-temporal-aq", "1")
    else:
        extra_args.extend(["-crf", str(crf)])

    return {
        "codec": codec_result,
        "encoder": codec_result,
        "preset": preset,
        "tune": tune_flag,
        "pix_fmt": pix_fmt,
        "extra_args": extra_args,
        "is_nvenc": is_nvenc,
    }


def get_encoding_config_from_dict(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize encoding config from a settings dict.

    Args:
        config: Settings dictionary that may contain encoding keys

    Returns:
        Normalized encoding config dict with defaults applied
    """
    result = DEFAULT_ENCODING_CONFIG.copy()

    if "codec" in config:
        result["codec"] = config["codec"]

    if "encoder" in config:
        result["encoder"] = config["encoder"]
    if "encoding_encoder" in config:
        result["encoder"] = config["encoding_encoder"]

    if "quality" in config:
        result["quality"] = config["quality"]
    if "encoding_quality" in config:
        result["quality"] = config["encoding_quality"]

    if "tune" in config:
        result["tune"] = config["tune"]
    if "encoding_tune" in config:
        result["tune"] = config["encoding_tune"]

    if "crf" in config:
        result["crf"] = int(config["crf"])
    if "output_crf" in config:
        result["crf"] = int(config["output_crf"])

    if "nvenc_lookahead_enabled" in config:
        result["nvenc_lookahead_enabled"] = config["nvenc_lookahead_enabled"]
    if "nvenc_lookahead" in config:
        result["nvenc_lookahead"] = int(config["nvenc_lookahead"])
    if "nvenc_spatial_aq" in config:
        result["nvenc_spatial_aq"] = config["nvenc_spatial_aq"]
    if "nvenc_temporal_aq" in config:
        result["nvenc_temporal_aq"] = config["nvenc_temporal_aq"]
    if "nvenc_aq_strength" in config:
        result["nvenc_aq_strength"] = int(config["nvenc_aq_strength"])

    return result
