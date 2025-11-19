import math
import torch

from flex_block_attn_cuda import attn_forward, attn_backward

def _flex_block_attn_forward(q, k, v, q_block_size, kv_block_size, block_mask, small_block_mask, use_small_block_mode):
    o, lse = attn_forward(q, k, v, q_block_size, kv_block_size, block_mask, small_block_mask, use_small_block_mode)
    return o, lse 

def _flex_block_attn_backward(q, k, v, o, lse, do, q_block_size, kv_block_size, block_mask, small_block_mask, use_small_block_mode):
    dq, dk, dv = attn_backward(q, k, v, o, lse, do, q_block_size, kv_block_size, block_mask, small_block_mask, use_small_block_mode)
    return dq, dk, dv

class FlexBlockAttnFunc(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, q_block_size, kv_block_size, block_mask, is_grad_enabled):
        assert q_block_size % 16 == 0, f"q_block_size ({q_block_size}) must be divisible by 16"
        assert kv_block_size % 64 == 0, f"kv_block_size ({kv_block_size}) must be divisible by 64"
        assert kv_block_size % q_block_size == 0 or q_block_size % kv_block_size == 0, f"kv_block_size ({kv_block_size}) must be divisible by q_block_size ({q_block_size}), or vice versa."
        _, _, q_seq_len, _ = q.shape
        _, _, kv_seq_len, _ = k.shape
        assert q_seq_len % 64 == 0, f"q_seq_len ({q_seq_len}) must be divisible by 64"
        assert kv_seq_len % 64 == 0, f"kv_seq_len ({kv_seq_len}) must be divisible by 64"

        if block_mask.dim() == 2:
            q_block_num, kv_block_num = block_mask.shape
        else:
            _, _, q_block_num, kv_block_num = block_mask.shape

        assert q_block_size * q_block_num == q_seq_len, f"q_block_size ({q_block_size}) * q_block_num ({q_block_num}) must equal q_seq_len ({q_seq_len}"
        assert kv_block_size * kv_block_num == kv_seq_len, f"kv_block_size ({kv_block_size}) * kv_block_num ({kv_block_num}) must equal kv_seq_len ({kv_seq_len}"

        is_grad = is_grad_enabled and any(
            x.requires_grad for x in [q, k, v]
        )

        if kv_block_size / q_block_size > 1:
            ratio = kv_block_size // q_block_size
            small_block_mask = block_mask

            if block_mask.dim() == 2:
                q_block_num, kv_block_num = block_mask.shape
                block_mask_shape = (q_block_num // ratio, ratio, kv_block_num)
            else:
                batch_size, head_num, q_block_num, kv_block_num = block_mask.shape
                block_mask_shape = (batch_size, head_num, q_block_num//ratio, ratio, kv_block_num)

            block_mask = torch.max(block_mask.reshape(block_mask_shape), dim=-2)[0]
            use_small_block_mode = True
        elif q_block_size / kv_block_size > 1:
            ratio = q_block_size // kv_block_size
            small_block_mask = block_mask

            if block_mask.dim() == 2:
                q_block_num, kv_block_num = block_mask.shape
                block_mask_shape = (q_block_num // ratio, ratio, kv_block_num)
            else:
                batch_size, head_num, q_block_num, kv_block_num = block_mask.shape
                block_mask_shape = (batch_size, head_num, q_block_num, kv_block_num // ratio, ratio)

            block_mask = torch.max(block_mask.reshape(block_mask_shape), dim=-1)[0]
            use_small_block_mode = True
        else:
            small_block_mask = block_mask
            use_small_block_mode = False

        o, lse = _flex_block_attn_forward(q, k, v, q_block_size, kv_block_size, block_mask, small_block_mask, use_small_block_mode)

        if is_grad:
            ctx.save_for_backward(q, k, v, block_mask, small_block_mask, o, lse)
            ctx.q_block_size = q_block_size
            ctx.kv_block_size = kv_block_size
            ctx.use_small_block_mode = use_small_block_mode

        return o

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, block_mask, small_block_mask, o, lse = ctx.saved_tensors
        q_block_size = ctx.q_block_size
        kv_block_size = ctx.kv_block_size
        use_small_block_mode = ctx.use_small_block_mode

        block_mask = block_mask.transpose(-1, -2).contiguous()
        small_block_mask = small_block_mask.transpose(-1, -2).contiguous()

        dq, dk, dv = _flex_block_attn_backward(q, k, v, o, lse, dout, q_block_size, kv_block_size, block_mask, small_block_mask, use_small_block_mode)
        return dq, dk, dv, None, None, None, None, None

def flex_block_attn_func(q, k, v, q_block_size, kv_block_size, block_mask):
    return FlexBlockAttnFunc.apply(q, k, v, q_block_size, kv_block_size, block_mask, torch.is_grad_enabled())

