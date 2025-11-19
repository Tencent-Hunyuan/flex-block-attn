import torch
import numpy as np
from flex_sta_ref import get_sliding_tile_attention_mask
from flex_block_attn import  flex_block_attn_func, create_sta_3d_mask
from flex_block_attn.utils import visualize_attn_mask
from torch.nn.attention.flex_attention import flex_attention
from tqdm import tqdm

flex_attention = torch.compile(flex_attention, dynamic=False)

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude

    return scaled_tensor.contiguous()

def perf_task(b, h, n, d, block_size, mean, std, num_iterations=1):
    print(f"seq_len:{n}, block_size:{block_size}")
    from tqdm import tqdm
    for step in range(num_iterations):
        torch.manual_seed(0)

        Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda').requires_grad_()
        K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda').requires_grad_()
        V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda').requires_grad_()
        dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
        
        #torch_mask, block_mask = create_mask(Q)
        import time
        torch.cuda.synchronize()
        #block_num = int(n / block_size)
        #block_mask = torch.ones((block_num, block_num), dtype=torch.bool, device=Q.device)
        #canvas_thw = (36, 48, 48)
        #canvas_thw = (18, 32, 32)
        #canvas_thw = (2 * 6, 3 * 8, 2 * 8)
        canvas_thw = (3 * 4, 3 * 4, 3 * 4)
        #tile_thw = (6, 8, 8)
        tile_thw = (4, 4, 4)
        #kernel_thw = (3, 3, 3)
        kernel_thw = (2, 2, 2)
        block_mask = create_sta_3d_mask(Q, K, canvas_thw, tile_thw, kernel_thw, 0)
        start1 = time.time()
        ptm_o = flex_block_attn_func(Q, K, V, block_size, block_mask)
        torch.cuda.synchronize()
        start2 = time.time()
        ptm_o.backward(dO)
        torch.cuda.synchronize()
        start3 = time.time()

        Q.grad.zero_()
        K.grad.zero_()
        V.grad.zero_()

        torch.cuda.synchronize()
        mask, image_mask = get_sliding_tile_attention_mask(kernel_thw, tile_thw, canvas_thw, 0, 'cuda', 0)
        print("mask", mask)
        start4 = time.time()
        #pt_o = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        pt_o = flex_attention(Q, K, V, block_mask=mask)
        torch.cuda.synchronize()
        start5 = time.time()
        pt_o.backward(dO)
        torch.cuda.synchronize()
        start6 = time.time()

        visualize_attn_mask(block_mask, save_path='pyb_sta_3d_mask.png')
        from attn_gym import visualize_attention_scores
        visualize_attention_scores(Q, K, mask_mod=image_mask, device="cuda", name="pyb_sta_3d_mask_torch.png")

        print(f"seq_len:{n}, block_size:{block_size}, step:{step}, full-attn ptm_fwd_utime:{start2-start1}, ptm_bwd_utime:{start3-start2}, torch_fwd_utime:{start5-start4}, torch_bwd_utime:{start6-start5}")

def run(b, h, d, mean, std):
    #seq_lengths = [82944]
    #seq_lengths = [18432]
    seq_lengths = [1728]
    #seq_lengths = [26 * 10368]
    #seq_lengths = [384 * 8]
    #seq_lengths = [384 * 2]

    #block_sizes = [384]
    block_sizes = [64]

    tk_avg_errors, tk_max_errors = [], []

    for sel_len in tqdm(seq_lengths, desc="seq_len"):
        for block_size in block_sizes:
            results = perf_task(b, h, sel_len, d, block_size, mean, std)

# Example usage
b, h, d = 1, 1, 128
mean = 1e-1
std = 10

run(b, h, d, mean, std)

