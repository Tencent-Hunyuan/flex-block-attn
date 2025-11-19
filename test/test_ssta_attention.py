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
    #pyb
    from flex_block_attn.ssta_attention import create_sta_3d_mask
    torch_mask, _ = create_sta_3d_mask(Q, K, canvas_thw, tile_thw, kernel_thw, return_torch_mask=True)
    pt_o = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=torch_mask)
    pt_o = untile(pt_o, canvas_thw, tile_thw)
    return pt_o

def check_correctness(b, h, n, d, canvas_thw, causal, mean, std, num_iterations=1, error_mode='all'):
    results = {
        'TK vs FLEX': {
            'sum_diff': 0,
            'sum_abs': 0,
            'max_diff': 0
        },
    }
    #kernel_thw_list = [(6, 1, 6), (6, 6, 1)]
    #kernel_thw_list = [(1, 1, 1), (2, 2, 2)]
    #kernel_thw_list = [(1, 1, 1)]
    kernel_thw_list = [(2, 2, 2)]
    #kernel_thw_list = [(8, 8, 8)]

    from tqdm import tqdm
    for kernel_thw in tqdm(kernel_thw_list):
        for _ in range(num_iterations):
            torch.manual_seed(0)

            Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda').requires_grad_()
            K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda').requires_grad_()
            V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda').requires_grad_()
            dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
           
            topk=2
            #tile_thw=(6, 8, 8)
            tile_thw=(2, 8, 8)

            sparse_type = 'sta'
            #sparse_type = 'ssta'
            #sparse_type = 'block_attn'
            tk_o = ssta_3d_attention(Q, K, V, canvas_thw, topk=topk, tile_thw=tile_thw, kernel_thw=kernel_thw, sparse_type=sparse_type)
            tk_o.backward(dO)
            tk_q_grad = Q.grad.detach().clone()
            tk_k_grad = K.grad.detach().clone()
            tk_v_grad = V.grad.detach().clone()

            Q.grad.zero_()
            K.grad.zero_()
            V.grad.zero_()

            pt_o = torch_attn(Q, K, V, canvas_thw, topk, tile_thw, kernel_thw)
            pt_o.backward(dO)
            pt_q_grad = Q.grad
            pt_k_grad = K.grad
            pt_v_grad = V.grad

            diff = pt_o - tk_o
            abs_diff = torch.abs(diff)
            results['TK vs FLEX']['sum_diff'] += torch.sum(abs_diff).item()
            results['TK vs FLEX']['max_diff'] = max(results['TK vs FLEX']['max_diff'], torch.max(abs_diff).item())

            #print("pyb tk_q_grad", tk_q_grad[:, :, 900:1000, 0], "pt_q_grad", pt_q_grad[:, :, 900:1000, 0])
            #print("pyb tk_q_grad", tk_q_grad, "pt_q_grad", pt_q_grad)
            diff_q_grad = pt_q_grad - tk_q_grad
            abs_diff_q_grad = torch.abs(diff_q_grad)
            #print("abs_diff_q_grad", abs_diff_q_grad[:, :, :, 0])
            #print("abs_diff_q_grad max", torch.max(abs_diff_q_grad[:, :, :, 0]))

            print("pyb tk_k_grad", tk_k_grad, "pt_k_grad", pt_k_grad)
            diff_k_grad = pt_k_grad - tk_k_grad
            abs_diff_k_grad = torch.abs(diff_k_grad)
            #print("abs_diff_k_grad", abs_diff_k_grad[:, :, :, 0])
            #print("abs_diff_k_grad max", torch.max(abs_diff_k_grad[:, :, :, 0]))
            
            diff_v_grad = pt_v_grad - tk_v_grad
            abs_diff_v_grad = torch.abs(diff_v_grad)

            torch.cuda.empty_cache()
        print("kernel_thw", kernel_thw)
        print("max_diff_o", torch.max(abs_diff).item())
        #print(pt_o.reshape(-1)[289988], tk_o.reshape(-1)[289988])
        #print("max_diff_o_index", torch.argmax(abs_diff).item())
        #print("max_diff_o_ref", torch.argmax(abs_diff/torch.abs(pt_o)).item())
        print("avg_diff_o_ref_mean", torch.mean(abs_diff/torch.abs(pt_o)).item())
        print("avg_diff_o", torch.sum(abs_diff).item() / abs_diff.numel())
        print("max_diff_q_grad", torch.max(abs_diff_q_grad).item())
        #print("max_diff_q_grad_index", torch.argmax(abs_diff_q_grad).item())
        #4893099
        #print(f"max_diff_q_grad_index_value torch:", pt_q_grad.reshape(-1)[4893099], "tk", tk_q_grad.reshape(-1)[4893099])
        print("avg_diff_q_grad", torch.sum(abs_diff_q_grad).item() / abs_diff_q_grad.numel())
        print("max_diff_k_grad", torch.max(abs_diff_k_grad).item())
        print("avg_diff_k_grad", torch.sum(abs_diff_k_grad).item() / abs_diff_k_grad.numel())
        print("max_diff_v_grad", torch.max(abs_diff_v_grad).item())
        print("avg_diff_v_grad", torch.sum(abs_diff_v_grad).item() / abs_diff_v_grad.numel())


    #total_elements = b * h * n * d * num_iterations * (1 if error_mode == 'output' else
    #                                                   3 if error_mode == 'backward' else 4) * len(kernel_size_ls)
    
    total_elements = b * h * n * d * num_iterations
    for name, data in results.items():
        avg_diff = data['sum_diff'] / total_elements
        max_diff = data['max_diff']
        results[name] = {'avg_diff': avg_diff, 'max_diff': max_diff}

    return results


def generate_error_graphs(b, h, d, causal, mean, std, error_mode='all'):
    #seq_lengths = [82944]
    #seq_lengths = [10368]
    #seq_lengths = [384 * 8]
    #seq_lengths = [(30 * 48 * 80, (30, 48, 80))]
    #seq_lengths = [(24 * 8 * 8, (24, 8, 8))]
    #seq_lengths = [(18 * 8 * 8, (18, 8, 8))]
   # seq_lengths = [(12 * 8 * 8, (12, 8, 8))]
    seq_lengths = [(18 * 24 * 32, (18, 24, 32))]

    tk_avg_errors, tk_max_errors = [], []

    for (n, canvas_thw) in tqdm(seq_lengths, desc="Generating error data"):
        results = check_correctness(b, h, n, d, canvas_thw, causal, mean, std, error_mode=error_mode)

        tk_avg_errors.append(results['TK vs FLEX']['avg_diff'])
        tk_max_errors.append(results['TK vs FLEX']['max_diff'])


# Example usage
b, h, d = 2, 2, 128
causal = False
mean = 1e-1
std = 10

for mode in ['output']:
    generate_error_graphs(b, h, d, causal, mean, std, error_mode=mode)

