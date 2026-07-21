"""Attention processor that routes large spatial self-attention through SageAttention.

SageAttention's quantized kernel is only a win for long sequences: on the
shapes DepthCrafter produces (640x384, 110-frame window) it is ~4x faster than
SDPA for the 3840-token spatial self-attention, but *slower* for the 110-token
temporal attention and unsupported for head dims outside {64, 96, 128}. The
processor therefore dispatches per call and falls back to SDPA everywhere else.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention

_logger = logging.getLogger(__name__)

try:
    from sageattention import sageattn

    SAGE_ATTENTION_AVAILABLE = True
except ImportError:
    sageattn = None
    SAGE_ATTENTION_AVAILABLE = False

# Below this sequence length the quantization overhead outweighs the kernel
# speedup (measured crossover on an RTX 5080 sits well under 512).
SAGE_MIN_SEQ_LEN = 512
_SAGE_HEAD_DIMS = (64, 96, 128)


def _attention(query, key, value, attention_mask):
    if (
        sageattn is not None
        and attention_mask is None
        and query.is_cuda
        and query.dtype in (torch.float16, torch.bfloat16)
        and query.shape[-1] in _SAGE_HEAD_DIMS
        and query.shape[-2] >= SAGE_MIN_SEQ_LEN
        and query.shape[-2] == key.shape[-2]
    ):
        return sageattn(query, key, value, tensor_layout="HND", is_causal=False)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
    )


class SageAttnProcessor2_0:
    """Drop-in replacement for diffusers' AttnProcessor2_0 with SageAttention dispatch."""

    def __init__(self):
        if not SAGE_ATTENTION_AVAILABLE:
            raise ImportError("SageAttnProcessor2_0 requires the 'sageattention' package.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        hidden_states = _attention(query, key, value, attention_mask)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
