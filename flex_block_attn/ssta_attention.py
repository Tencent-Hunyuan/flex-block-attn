import math
import torch
import numpy as np
from einops import rearrange
import os
import random
from functools import lru_cache

from flex_block_attn.utils import visualize_attn_mask
from .flex_block_attn_interface import flex_block_attn_func
from torch import distributed as dist

DEBUG=int(os.environ.get("DEBUG", 0))
def info(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
def random_info(*args, **kwargs):
    if int(os.environ["RANK"]) <= 0 and (random.random() < 0.01):
        print(*args, **kwargs)

class _AllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, dim, group):
        ctx.dim = dim
        ctx.group = group
        world_size = dist.get_world_size(group)
        input_size = list(input_.size())

        sizes = [None] * world_size
        dist.all_gather_object(sizes, input_.shape, group)

        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty(sizes[i], dtype=input_.dtype, device=input_.device) for i in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        world_size = dist.get_world_size(group)
        global_rank = dist.get_rank()
        rank = dist.get_group_rank(group, global_rank)
        dim = ctx.dim
        input_size = ctx.input_size

        # sizes = [input_size] * world_size

        sizes = [None] * world_size
        dist.all_gather_object(sizes, input_size, group=group)

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None, None
def all_gather(input_: torch.Tensor, dim: int = 1, group=None):
    return _AllGather.apply(input_, dim, group)
    
def tile(x, canvas_thw, tile_thw, sp_size=1):
    b, h, s, d = x.shape
    t, h, w = canvas_thw
    assert t * h * w == s, f"t:{t} * h:{h} * w:{w} == s:{s}"

    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    n_t = int(t/tile_t_dim)
    n_h = int(h/tile_h_dim)
    n_w = int(w/tile_w_dim)
    x = rearrange(x, "b head (sp t h w) d -> b head (t sp h w) d", sp=sp_size, t=t // sp_size, h=h, w=w)
    return rearrange(x,
                     "b h (n_t ts_t n_h ts_h n_w ts_w) d -> b h (n_t n_h n_w ts_t ts_h ts_w) d",
                     n_t=n_t,
                     n_h=n_h,
                     n_w=n_w,
                     ts_t=tile_t_dim,
                     ts_h=tile_h_dim,
                     ts_w=tile_w_dim)

def untile(x, canvas_thw, tile_thw, sp_size=1):
    t, h, w = canvas_thw

    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    n_t = int(t/tile_t_dim)
    n_h = int(h/tile_h_dim)
    n_w = int(w/tile_w_dim)

    x = rearrange(x,
                  "b h (n_t n_h n_w ts_t ts_h ts_w) d -> b h (n_t ts_t n_h ts_h n_w ts_w) d",
                  n_t=n_t,
                  n_h=n_h,
                  n_w=n_w,
                  ts_t=tile_t_dim,
                  ts_h=tile_h_dim,
                  ts_w=tile_w_dim)
    return rearrange(x, "b head (t sp h w) d -> b head (sp t h w) d", sp=sp_size, t=t // sp_size, h=h, w=w)

def get_tile_t_h_w(tile_id, tile_thw_dim):
    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw_dim
    tile_t = tile_id // (tile_h_dim * tile_w_dim)
    tile_h = (tile_id % (tile_h_dim * tile_w_dim)) // tile_w_dim
    tile_w = tile_id % tile_w_dim

    return tile_t, tile_h, tile_w

def create_moba_3d_mask(q, k, canvas_thw, topk, tile_thw, kernel_thw, text_block_num=0, add_text_mask=False, 
                        return_torch_mask=False, threshold=0.0, mask_share_within_head=True):
    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    kernel_t, kernel_h, kernel_w = kernel_thw

    seq_len = q.size(2)
    block_size = np.prod(tile_thw)
    block_num = int(seq_len / block_size)
    info("block_size", block_size, "block_num", block_num)

    ## moba
    image_shape = canvas_thw
    block_shape = tile_thw
    batch_size, num_heads, seq_len, head_dim  = k.shape
    num_blocks_t = np.ceil(image_shape[0] / block_shape[0]).astype(int)
    num_blocks_h = np.ceil(image_shape[1] / block_shape[1]).astype(int)
    num_blocks_w = np.ceil(image_shape[2] / block_shape[2]).astype(int)
    num_blocks = num_blocks_t * num_blocks_h * num_blocks_w
    info(f"image_shape:{image_shape}, block_shape:{block_shape}, num_blocks_t:{num_blocks_t}, num_blocks_h:{num_blocks_h}, num_blocks_w:{num_blocks_w}")

    def get_block_avg_feat(x: torch.Tensor) -> torch.Tensor:
        x_block_means = x.view(batch_size, num_heads, num_blocks_t, num_blocks_h, num_blocks_w,
                                               block_shape[0], block_shape[1], block_shape[2], head_dim)
        x_block_means = x_block_means.mean(dim=(-2, -3, -4)).view(batch_size, num_heads, -1, head_dim)
        return x_block_means

    k_block_means = get_block_avg_feat(k)
    q = get_block_avg_feat(q)
    q = q.type(torch.float32)
    k_block_means = k_block_means.type(torch.float32)  # float logit for better gate logit perception

    if mask_share_within_head:
        q = q.mean(dim=1, keepdim=True)
        k_block_means = k_block_means.mean(dim=1, keepdim=True)

    gate = torch.einsum("bhsd,bhkd->bhsk", q, k_block_means)
    q = q.type_as(k)
    
    topk = min(topk, gate.size(-1))  # 确保 topk 不超过 block_num
    
    if threshold > 0.0:
        assert mask_share_within_head, "mask_share_within_head must be True when threshold > 0.0" #实验发现分head的累积和阈值判断不准确
        gate_ = torch.softmax(gate, dim=-1)
        sorted_gate, sorted_indices = torch.sort(gate_, dim=-1, descending=True)
        cum_scores = torch.cumsum(sorted_gate, dim=-1)
        above_threshold = cum_scores >= threshold
        has_any_above = above_threshold.any(dim=-1, keepdim=True)
        dynamic_topk = above_threshold.int().argmax(dim=-1, keepdim=True) + 1 #找到第一个超过阈值的位置，加1得到需要选择的block数量
        dynamic_topk = torch.where(has_any_above, dynamic_topk,
                                   torch.full_like(dynamic_topk, topk)) # 不满足全选
                                   #torch.ones_like(dynamic_topk))

        # 限制最大 k 值
        dynamic_topk = torch.clamp(dynamic_topk, max=topk)
        # 生成固定形状的索引并将top1的索引填充到未满足阈值的部分
        index_shape = gate.size()[:-1] + (topk,)
        indices = torch.arange(topk, device=gate.device).expand(index_shape)
        mask = (indices < dynamic_topk).int() 
        top_block_indices = torch.gather(sorted_indices, -1, indices) * mask + (1 - mask) * sorted_indices[..., 0:1] #  取top1填充到topk以保证同大小
        
        
        with torch.no_grad():
            top_block_scores = torch.gather(sorted_gate, -1, indices) * mask 
            top_block_scores = top_block_scores.sum(dim=-1, keepdim=True)
            random_info(f"""
                        topk_param={topk}
                        head_first_topk={dynamic_topk[0,0,[0,dynamic_topk.size(-1)//2, -1],:].detach().cpu().tolist()}
                        head_first_block_scores={top_block_scores[0,0,[0,top_block_scores.size(-2)//2, -1],:].detach().cpu().tolist()}
                        head_last_topk={dynamic_topk[0,-1,[0,dynamic_topk.size(-1)//2, -1],:].detach().cpu().tolist()}
                        head_last_top_block_scores={top_block_scores[0,-1,[0,top_block_scores.size(-2)//2, -1],:].detach().cpu().tolist()}
                        """)
    else:
        _, top_block_indices = gate.topk(k=topk, dim=-1)  # 取索引，形状为 [b, h, s_block, topk]
    
    # 保留head维度动态构建mask
    assert top_block_indices.size(0)==1, "top_block_indices batch size must be 1"
    top_block_indices = top_block_indices.squeeze(0)  # 仅移除batch维度 [h, s_block, topk]
    
    # 创建3D mask模板 (heads, block_num, block_num)
    gate_idx_mask = torch.zeros(
        (top_block_indices.size(0), block_num, block_num), 
        dtype=torch.bool,
        device=q.device
    )
    
    # 使用dim=0遍历每个注意力头
    for head_idx in range(top_block_indices.size(0)):
        # 为每个头构建索引矩阵
        head_mask = torch.zeros(
            (block_num, block_num), 
            dtype=torch.bool, 
            device=q.device
        )
        # 在当前头填充索引
        head_mask.scatter_(
            dim=-1, 
            index=top_block_indices[head_idx], 
            value=True
        )
        gate_idx_mask[head_idx] = head_mask

    if text_block_num > 0:
        pad_block_num = block_num + text_block_num 
        moba_3d_mask = torch.full(
            (gate_idx_mask.size(0), pad_block_num, pad_block_num), 
            False, 
            dtype=torch.bool, 
            device=gate_idx_mask.device
        )
        moba_3d_mask[:, :block_num, :block_num] = gate_idx_mask
        if add_text_mask:
            moba_3d_mask[:, :, -text_block_num:] = True
            moba_3d_mask[:, -text_block_num:, :] = True
    else:
        moba_3d_mask = gate_idx_mask

    return moba_3d_mask

@lru_cache(maxsize=4096)
def create_sta_3d_mask_(seq_len, tile_thw, kernel_thw, text_block_num=0, return_torch_mask=False):
    seq_len = int(seq_len)
    tile_thw = tuple(map(int, tile_thw.split('_')))
    kernel_thw = tuple(map(int, kernel_thw.split('_')))

    tile_t_dim, tile_h_dim, tile_w_dim = tile_thw
    kernel_t, kernel_h, kernel_w = kernel_thw

    #seq_len = q.size(2)
    block_size = np.prod(tile_thw)
    block_num = int(seq_len / block_size)
    info("block_size", block_size, "block_num", block_num)

    block_mask = np.full((block_num + text_block_num, block_num + text_block_num), False, dtype=bool)
    if return_torch_mask:
        torch_mask = np.full((seq_len + text_block_num * block_size, seq_len + text_block_num * block_size), False, dtype=bool)

    # sta
    for i in range(block_num + text_block_num):
        for j in range(block_num + text_block_num):
            q_t_tile, q_h_tile, q_w_tile = get_tile_t_h_w(i, tile_thw)
            kv_t_tile, kv_h_tile, kv_w_tile = get_tile_t_h_w(j, tile_thw)

            kernel_center_t = np.clip(q_t_tile, kernel_t // 2, (tile_t_dim - 1) - kernel_t // 2)
            kernel_center_h = np.clip(q_h_tile, kernel_h // 2, (tile_h_dim - 1) - kernel_h // 2)
            kernel_center_w = np.clip(q_w_tile, kernel_w // 2, (tile_w_dim - 1) - kernel_w // 2)

            time_mask = abs(kernel_center_t - kv_t_tile) <= kernel_t // 2
            hori_mask = abs(kernel_center_h - kv_h_tile) <= kernel_h // 2
            vert_mask = abs(kernel_center_w - kv_w_tile) <= kernel_w // 2

            if (time_mask and hori_mask and vert_mask) or i >= block_num or j >= block_num:
                # torch_mask
                block_mask[i, j] = True

                # torch_mask
                if return_torch_mask:
                    for x in range(i*block_size, (i+1) * block_size):
                        for y in range(j*block_size, (j+1) * block_size):
                            torch_mask[x, y] = True

    block_mask = torch.tensor(block_mask, dtype=torch.bool)
    if return_torch_mask:
        torch_mask = torch.tensor(torch_mask, dtype=torch.bool)

    info("block_mask", block_mask)
    if return_torch_mask:
        #pyb
        #info("torch_mask", torch_mask)
        #visualize_attn_mask(torch_mask, save_path='torch_attention_mask.png')
        return torch_mask, block_mask
    else:
        return block_mask

def create_sta_3d_mask(q, k, canvas_thw, topk, tile_thw, kernel_thw, text_block_num=0, return_torch_mask=False):
    block_mask = create_sta_3d_mask_(str(q.size(2)),
                               "_".join([str(x) for x in tile_thw]),
                               "_".join([str(x) for x in kernel_thw]),
                               text_block_num, return_torch_mask)
    return block_mask.to(q.device)

def create_ssta_3d_mask(q, k, canvas_thw, topk, tile_thw, kernel_thw, text_block_num=0, return_torch_mask=False, 
                        threshold=0.0, 
                        text_mask=None, 
                        sp_size=1, sp_rank=0, sp_group=None,
                        mask_share_within_head=True):
    sp_enabled = sp_size > 1
    import time
    start1 = time.time()
    sta_3d_mask = create_sta_3d_mask(q, k, canvas_thw, topk, tile_thw, kernel_thw, text_block_num, return_torch_mask)

    if sp_enabled:
        assert sp_group is not None
        q = all_gather(q, dim=1, group=sp_group)
        k = all_gather(k, dim=1, group=sp_group)

    start2 = time.time()
    moba_3d_mask = create_moba_3d_mask(q, k, canvas_thw, topk, tile_thw, kernel_thw, text_block_num, return_torch_mask, 
                                       threshold=threshold, mask_share_within_head=mask_share_within_head)
    start3 = time.time()
    info(f"pyb sta_3d_mask use time:{start2 - start1}, moba_3d_mask use time:{start3 - start2}")



    ssta_3d_mask = torch.logical_or(sta_3d_mask.unsqueeze(0), moba_3d_mask)
    assert len(ssta_3d_mask.size()) == 3, "ssta_3d_mask should be 3D"
    
    # set text block mask
    if text_mask is not None:
        block_size = np.prod(tile_thw)
        seq_len = q.size(2)
        block_num = int(seq_len / block_size)
        text_mask_index = torch.ceil(text_mask.sum() / block_size).long()
        text_mask_index = text_mask_index.clamp(min=1).item()
        assert ssta_3d_mask.shape[-1] == ssta_3d_mask.shape[-2], "ssta_3d_mask should be square"
        
        pad_start_index = block_num + text_mask_index
        ssta_3d_mask[:, pad_start_index:, :] = False
        ssta_3d_mask[:, :, pad_start_index:] = False
        eye_mask = torch.eye(ssta_3d_mask.shape[1]-pad_start_index, dtype=torch.bool, device=ssta_3d_mask.device).unsqueeze(0)
        ssta_3d_mask[:, pad_start_index:, pad_start_index:] = ssta_3d_mask[:, pad_start_index:, pad_start_index:] | eye_mask
            
        random_info(f"seq_len={seq_len}, \
                      ssta_3d_mask.shape={ssta_3d_mask.shape}, \
                      block_num={block_num}, \
                      text_mask_index={text_mask_index}, \
                      text_mask_index_abs={text_mask_index+block_num}")
        
    info("pyb ssta_3d_mask", ssta_3d_mask)
    return ssta_3d_mask

def ssta_3d_attention(all_q, all_k, all_v, canvas_thw, topk=1, tile_thw=(6, 8, 8), kernel_thw=(1, 1, 1), 
	                 text_len=0, sparse_type='ssta', threshold=0.0,
	                 pad_type="zero",
                     text_mask=None,
                     mask_share_within_head=True,
                     sp_size=1, sp_rank=0, sp_group=None):
    info("pyb canvas_thw", canvas_thw, "tile_thw", tile_thw, "all_q.shape", all_q.shape, "kernel_thw", kernel_thw,"text_len", text_len)
    assert pad_type in ["zero", "repeat"]
    if text_len > 0:
        image_q = all_q[:, :, :-text_len, :]
        image_k = all_k[:, :, :-text_len, :]
        image_v = all_v[:, :, :-text_len, :]

        text_q = all_q[:, :, -text_len:, :]
        text_k = all_k[:, :, -text_len:, :]
        text_v = all_v[:, :, -text_len:, :]
    else:
        image_q = all_q
        image_k = all_k
        image_v = all_v

    b, hd, s, d = image_q.shape
    t, h, w = canvas_thw
    assert t * h * w == s, f"t:{t} * h:{h} * w:{w} != s:{s}"
    tile_t, tile_h, tile_w = tile_thw
    block_size = np.prod(tile_thw)

    import time
    start1 = time.time()
    need_pad = False
    if t % tile_t != 0 or h % tile_h !=0 or w % tile_w != 0:
        need_pad = True
        pad_image_q = image_q.reshape(b, hd, t, h, w, d)
        pad_image_k = image_k.reshape(b, hd, t, h, w, d)
        pad_image_v = image_v.reshape(b, hd, t, h, w, d)

        pad_t = 0 if t % tile_t == 0 else tile_t - t % tile_t
        if pad_t > 0:
            t = t + pad_t
            repeat_q = pad_image_q[:, :, -1:, :, :, :].expand(-1, -1, pad_t, -1, -1, -1)
            repeat_k = pad_image_k[:, :, -1:, :, :, :].expand(-1, -1, pad_t, -1, -1, -1)
            repeat_v = pad_image_v[:, :, -1:, :, :, :].expand(-1, -1, pad_t, -1, -1, -1)
            if pad_type == "zero":
                repeat_q = torch.zeros_like(repeat_q)
                repeat_k = torch.zeros_like(repeat_k)
                repeat_v = torch.zeros_like(repeat_v)
            pad_image_q = torch.cat([pad_image_q, repeat_q], dim=2)
            pad_image_k = torch.cat([pad_image_k, repeat_k], dim=2)
            pad_image_v = torch.cat([pad_image_v, repeat_v], dim=2)

        pad_h = 0 if h % tile_h == 0 else tile_h - h % tile_h
        if pad_h > 0:
            h = h + pad_h
            repeat_q = pad_image_q[:, :, :, -1:, :, :].expand(-1, -1, -1, pad_h, -1, -1)
            repeat_k = pad_image_k[:, :, :, -1:, :, :].expand(-1, -1, -1, pad_h, -1, -1)
            repeat_v = pad_image_v[:, :, :, -1:, :, :].expand(-1, -1, -1, pad_h, -1, -1)
            if pad_type == "zero":
                repeat_q = torch.zeros_like(repeat_q)
                repeat_k = torch.zeros_like(repeat_k)
                repeat_v = torch.zeros_like(repeat_v)
            pad_image_q = torch.cat([pad_image_q, repeat_q], dim=3)
            pad_image_k = torch.cat([pad_image_k, repeat_k], dim=3)
            pad_image_v = torch.cat([pad_image_v, repeat_v], dim=3)

        pad_w = 0 if w % tile_w == 0 else tile_w - w % tile_w
        if pad_w > 0:
            w = w + pad_w
            repeat_q = pad_image_q[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, pad_w, -1)
            repeat_k = pad_image_k[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, pad_w, -1)
            repeat_v = pad_image_v[:, :, :, :, -1:, :].expand(-1, -1, -1, -1, pad_w, -1)
            if pad_type == "zero":
                repeat_q = torch.zeros_like(repeat_q)
                repeat_k = torch.zeros_like(repeat_k)
                repeat_v = torch.zeros_like(repeat_v)
            pad_image_q = torch.cat([pad_image_q, repeat_q], dim=4)
            pad_image_k = torch.cat([pad_image_k, repeat_k], dim=4)
            pad_image_v = torch.cat([pad_image_v, repeat_v], dim=4)

        image_q = pad_image_q.reshape(b, hd, -1, d)
        image_k = pad_image_k.reshape(b, hd, -1, d)
        image_v = pad_image_v.reshape(b, hd, -1, d)

        canvas_thw = (t, h, w)
        info("pyb pad canvas_thw", canvas_thw, "image_q.shape", image_q.shape)

    need_pad_text = False
    text_block_num = math.ceil(text_len / block_size)
    text_target_size = text_block_num * block_size
    if text_len % block_size > 0:
        need_pad_text = True
        text_pad_size = text_target_size - text_len

        pad_text_q = text_q[:, :, -1, :].unsqueeze(2).expand(-1, -1, text_pad_size, -1)
        pad_text_k = text_k[:, :, -1, :].unsqueeze(2).expand(-1, -1, text_pad_size, -1)
        pad_text_v = text_v[:, :, -1, :].unsqueeze(2).expand(-1, -1, text_pad_size, -1)

        text_q = torch.cat([text_q, pad_text_q], dim=2)
        text_k = torch.cat([text_k, pad_text_k], dim=2)
        text_v = torch.cat([text_v, pad_text_v], dim=2)

        info("pyb text_q", text_q.shape, "text_pad_size", text_pad_size)
    #b, h, s, d = q.shape
    #t, h, w = canvas_thw
    #assert t * h * w == s, f"t:{t} * h:{h} * w:{w} != s:{s}"
    #assert t % tile_thw[0] == 0, f"t:{t} % tile_thw[0]:{tile_thw[0]} == 0"
    #assert h % tile_thw[1] == 0, "t % tile_thw[1] == 0"
    #assert w % tile_thw[2] == 0, "t % tile_thw[2] == 0"
    start2 = time.time()
    image_q = tile(image_q, canvas_thw, tile_thw)
    image_k = tile(image_k, canvas_thw, tile_thw)
    image_v = tile(image_v, canvas_thw, tile_thw)
    
    if text_len > 0:
        q = torch.cat([image_q, text_q], dim=2)
        k = torch.cat([image_k, text_k], dim=2)
        v = torch.cat([image_v, text_v], dim=2)
    else:
        q = image_q
        k = image_k
        v = image_v    

    if sparse_type == 'sta':
        assert text_mask is None, "text_mask do not support in sta sparse_type"
        block_mask = create_sta_3d_mask(image_q, image_k, canvas_thw, topk, tile_thw, kernel_thw, text_block_num)
        o = flex_block_attn_func(q, k, v, block_size, block_mask)
    elif sparse_type == 'block_attn':
        image_q_list = torch.split(image_q, 1, dim=0)
        image_k_list = torch.split(image_k, 1, dim=0)
        image_v_list = torch.split(image_v, 1, dim=0)

        mask_list = []
        for i in range(b):
            block_mask = create_moba_3d_mask(image_q_list[i], image_k_list[i], canvas_thw, topk, tile_thw, kernel_thw, text_block_num, 
                                                add_text_mask=True,
                                                mask_share_within_head=mask_share_within_head)
            mask_list.append(block_mask)
        block_mask = torch.stack(mask_list, dim=0)
        o = flex_block_attn_func(q, k, v, block_size, block_mask)

    elif sparse_type == 'ssta':
        image_q_list = torch.split(image_q, 1, dim=0)
        image_k_list = torch.split(image_k, 1, dim=0)
        image_v_list = torch.split(image_v, 1, dim=0)

        mask_list = []
        for i in range(b):
            block_mask = create_ssta_3d_mask(
                image_q_list[i],
                image_k_list[i],
                canvas_thw,
                topk,
                tile_thw,
                kernel_thw,
                text_block_num,
                threshold=threshold,
                text_mask=text_mask[i] if text_mask is not None else None,
                sp_size=sp_size, sp_rank=sp_rank, sp_group=sp_group,
                mask_share_within_head=mask_share_within_head
            )
            mask_list.append(block_mask)
        block_mask = torch.stack(mask_list, dim=0)
        if mask_share_within_head:
            block_mask = block_mask.unsqueeze(1)  # [b, 1, s_block, s_block]
        o = flex_block_attn_func(q, k, v, block_size, block_mask)
    else:
        raise Exception(f"unsurpport sparse_type:{sparse_type}")

    start3 = time.time()
    
    #info("pyb o.shape", o.shape)

    if text_len > 0:
        image_o = o[:, :, :-text_target_size, :]
        if need_pad_text:
            text_o = o[:, :, -text_target_size : -text_pad_size, :]
        else:
            text_o = o[:, :, -text_target_size:, :]
    else:
        image_o = o

    image_o = untile(image_o, canvas_thw, tile_thw)

    if need_pad:
        # 去掉o pad
        unpad_image_o = image_o.reshape(b, hd, t, h, w, d)

        if pad_t > 0: 
            unpad_image_o = unpad_image_o[:, :, :-pad_t, :, :, :]
        if pad_h > 0:
            unpad_image_o = unpad_image_o[:, :, :, :-pad_h, :, :]
        if pad_w > 0:
            unpad_image_o = unpad_image_o[:, :, :, :, :-pad_w, :]

        image_o = unpad_image_o.reshape(b, hd, -1, d)

    if text_len > 0:
        o = torch.cat([image_o, text_o], dim=2)
    else:
        o = image_o
    
    start4 = time.time() 
    info(f"pyb pad_utime:{start2-start1}, mask_attn_utime:{start3-start2}, post_utime:{start4-start3}")

    return o
