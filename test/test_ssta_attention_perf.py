import torch
from tqdm import tqdm

from flex_block_attn.ssta_attention import tile, untile
from flex_block_attn import create_ssta_3d_mask, ssta_3d_attention

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude

    return scaled_tensor.contiguous()

def torch_attn(Q, K, V, canvas_thw, topk, tile_thw, kernel_thw):
    Q = tile(Q, canvas_thw, tile_thw)
    K = tile(K, canvas_thw, tile_thw)
    V = tile(V, canvas_thw, tile_thw)
    torch_mask, _ = create_ssta_3d_mask(Q, K, topk, tile_thw, kernel_thw, return_torch_mask=True)
    pt_o = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=torch_mask)
    pt_o = untile(pt_o, canvas_thw, tile_thw)
    return pt_o

def check_correctness(b, h, n, d, canvas_thw, text_len, causal, mean, std, num_iterations=3, error_mode='all'):
    results = {
        'TK vs FLEX': {
            'sum_diff': 0,
            'sum_abs': 0,
            'max_diff': 0
        },
    }
    #kernel_thw_list = [(6, 1, 6), (6, 6, 1)]
    #kernel_thw_list = [(1, 1, 1), (2, 2, 2)]
    #kernel_thw_list = [(2, 3, 4)]
    kernel_thw_list = [(10, 10, 10)]

    from tqdm import tqdm
    for kernel_thw in tqdm(kernel_thw_list):
        for step in range(num_iterations):
            torch.manual_seed(0)

            Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda').requires_grad_()
            K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda').requires_grad_()
            V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda').requires_grad_()
            dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
           
            topk=10
            #kernel_thw=(1, 1, 1)
            tile_thw=(6, 8, 8)
            import time
            torch.cuda.synchronize()
            start1 = time.time()
            #sta/block_attn/ssta
            #sparse_type = 'ssta'
            sparse_type = 'sta'
            #sparse_type = 'block_attn'
            tk_o = ssta_3d_attention(Q, K, V, canvas_thw, topk=topk, tile_thw=tile_thw, kernel_thw=kernel_thw, text_len=text_len, sparse_type=sparse_type)
            torch.cuda.synchronize()
            start2 = time.time()
            print("tk_o", tk_o.shape, "dO", dO.shape)
            tk_o.backward(dO)
            torch.cuda.synchronize()
            start3 = time.time()



            torch.cuda.synchronize()
            start4 = time.time()

            start5 = time.time()
            #result.backward(dO_fa)
            torch.cuda.synchronize()
            start6 = time.time()

            print(f"step:{step}, ssta_3d_attention fwd_utime:{start2-start1}, bwd_utime:{start3-start2}, fa_fwd_utime:{start5-start4}, fa_bwd_utime:{start6-start5}")
    return results


def generate_error_graphs(b, h, d, causal, mean, std, error_mode='all'):
    #seq_lengths = [82944]
    #seq_lengths = [10368]
    #seq_lengths = [384 * 8]
    #seq_lengths = [(30 * 48 * 80, (30, 48, 80))]
    #seq_lengths = [(12 * 8 * 8, (12, 8, 8))]
    #720p: 17x78x44
    seq_lengths = [(17 * 78 * 44, (17, 78, 44), 0)]
    #seq_lengths = [(17 * 78 * 44 + 500, (17, 78, 44), 500)]
    #360p: 17x39x22
    #seq_lengths = [(17 * 39 * 22 + 50, (17, 39, 22), 50)]
    #seq_lengths = [(18 * 48 * 48, (18, 48, 48))]

    tk_avg_errors, tk_max_errors = [], []

    for (n, canvas_thw, text_len) in tqdm(seq_lengths, desc="Generating error data"):
        results = check_correctness(b, h, n, d, canvas_thw, text_len, causal, mean, std, error_mode=error_mode)

# Example usage
b, h, d = 1, 24, 128
causal = False
mean = 1e-1
std = 10

for mode in ['output']:
    generate_error_graphs(b, h, d, causal, mean, std, error_mode=mode)

