import torch
from flex_block_attn import  flex_block_attn_func
from tqdm import tqdm
import numpy as np

import pytest

def create_mask(Q, q_block_size, kv_block_size):
    seq_len = Q.size(2)
    mask_block_size = 2
    q_block_num = int(seq_len / q_block_size)
    kv_block_num = int(seq_len / kv_block_size)
    block_mask = np.full((q_block_num, kv_block_num), False, dtype=bool)
    mask = np.full((seq_len, seq_len), False, dtype=bool)

    for i in range(q_block_num):
        for j in range(kv_block_num):
            #if abs(j - i) < mask_block_size or j ==1:
            #if abs(j - i) < mask_block_size or (j ==0 and i ==1):
            #if abs(j - i) < mask_block_size or (j ==1 and i ==0):
            #if abs(j - i) < mask_block_size or j==0 or (i==3 and j==1):

            #if j==1 or (i==0 and j==0):
            #if j==0 or (i==0 and j==1):
            if abs(j - i) < mask_block_size or j==4:

            #if abs(j - i) < mask_block_size or (j==1 and i!=3) or (j==0 and i==3):
            #if (i != 4 or j!=0) and (i < 16 or j !=1):
            #if True:
            #if i < 4 or i > 8 or j < 1 or j > 1:
            #if i < 4 or i > 8 or j < 1 or i==7:
            #if j == 1 or (i > 4 and i < 7):
            #if i < 4 or j== 1:
            #if j==0 or (j==1 and i==0):
            #if True:
            #if i == 0 or j == 0:
            #if i != 0 or j != 0:
                block_mask[i, j] = True
                for x in range(i*q_block_size, (i+1) * q_block_size):
                    for y in range(j*kv_block_size, (j+1) * kv_block_size):
                        mask[x, y] = True
    block_mask = torch.tensor(block_mask, dtype=torch.bool, device=Q.device)
    torch_mask = torch.tensor(mask, dtype=torch.bool, device=Q.device)
    print("block_mask", block_mask)
    #visualize_attn_mask(mask)

    return torch_mask, block_mask

def torch_attn_test(Q, K, V, mask):
    output = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

    return output

def flex_block_attn_test(Q, K, V, q_block_size, kv_block_size, block_mask):
    o = flex_block_attn_func(Q, K, V, q_block_size, kv_block_size, block_mask)

    return o

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)

    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude

    return scaled_tensor.contiguous()

@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len", [384 * 2, 384 * 4, 384 * 8])
@pytest.mark.parametrize("head_num", [1, 2, 4])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("q_block_size", [16, 32, 64])
@pytest.mark.parametrize("kv_block_size", [64, 128, 256, 384])
def test_flex_block_attn(batch_size, seq_len, head_num, head_dim, q_block_size, kv_block_size):
    torch.manual_seed(0)

    mean = 1e-1
    std = 10
    Q = generate_tensor((batch_size, head_num, seq_len, head_dim), mean, std, torch.bfloat16, 'cuda').requires_grad_()
    K = generate_tensor((batch_size, head_num, seq_len, head_dim), mean, std, torch.bfloat16, 'cuda').requires_grad_()
    V = generate_tensor((batch_size, head_num, seq_len, head_dim), mean, std, torch.bfloat16, 'cuda').requires_grad_()
    dO = generate_tensor((batch_size, head_num, seq_len, head_dim), mean, std, torch.bfloat16, 'cuda')

    torch_mask, block_mask = create_mask(Q, q_block_size, kv_block_size)
    print("block_mask1 shape", block_mask.shape)
    block_mask = torch.stack([block_mask] * head_num, dim=0)
    block_mask = torch.stack([block_mask] * batch_size, dim=0).contiguous()
    print("block_mask2 shape", block_mask.shape)
    import time
    start1 = time.time()
    tk_o = flex_block_attn_test(Q, K, V, q_block_size, kv_block_size, block_mask)
    start2 = time.time()
    tk_o.backward(dO)
    start3 = time.time()
    print(f"flex_block_attn fwd_utime:{start2 - start1}, bwd_utime:{start3-start2}")
    tk_q_grad = Q.grad.detach().clone()
    tk_k_grad = K.grad.detach().clone()
    tk_v_grad = V.grad.detach().clone()

    Q.grad.zero_()
    K.grad.zero_()
    V.grad.zero_()

    pt_o = torch_attn_test(Q, K, V, torch_mask)
    pt_o.backward(dO)
    pt_q_grad = Q.grad
    pt_k_grad = K.grad
    pt_v_grad = V.grad

    diff = pt_o - tk_o
    print("pyb tk_o", tk_o[:, :, :, 0], "pt_o", pt_o[:, :, :, 0])
    abs_diff = torch.abs(diff)
    print("abs_diff", abs_diff[:, :, :, 0])
    print("abs_o max", torch.max(abs_diff[:, :, :, 0]))
    print("abs_o max argmax()", abs_diff[:, :, :, 0].argmax())

    print("pyb tk_q_grad", tk_q_grad[:, :, :, 0], "pt_q_grad", pt_q_grad[:, :, :, 0])
    diff_q_grad = pt_q_grad - tk_q_grad
    abs_diff_q_grad = torch.abs(diff_q_grad)
    print("abs_diff_q_grad", abs_diff_q_grad[:, :, :, 0])
    print("abs_diff_q_grad max", torch.max(abs_diff_q_grad[:, :, :, :]))
    print("abs_diff_q_grad max argmax()", abs_diff_q_grad[:, :, :, :].argmax())

    diff_k_grad = pt_k_grad - tk_k_grad
    abs_diff_k_grad = torch.abs(diff_k_grad)

    diff_v_grad = pt_v_grad - tk_v_grad
    abs_diff_v_grad = torch.abs(diff_v_grad)

    print("max_diff_o", torch.max(abs_diff).item())
    print("avg_diff_o", torch.sum(abs_diff).item() / abs_diff.numel())

    print("max_diff_q_grad", torch.max(abs_diff_q_grad).item())
    print("avg_diff_q_grad", torch.sum(abs_diff_q_grad).item() / abs_diff_q_grad.numel())
    print("max_diff_k_grad", torch.max(abs_diff_k_grad).item())
    print("avg_diff_k_grad", torch.sum(abs_diff_k_grad).item() / abs_diff_k_grad.numel())
    print("max_diff_v_grad", torch.max(abs_diff_v_grad).item())
    print("avg_diff_v_grad", torch.sum(abs_diff_v_grad).item() / abs_diff_v_grad.numel())

    torch.cuda.empty_cache()
    assert torch.allclose(tk_o, pt_o, rtol=1e-02, atol=5e-03)
    #assert torch.allclose(tk_q_grad, pt_q_grad, rtol=5e-03, atol=5e-03)

test_flex_block_attn(batch_size=1, seq_len=128, head_num=1, head_dim=128, q_block_size=16, kv_block_size=64)
#test_flex_block_attn(batch_size=1, seq_len=3072, head_num=1, head_dim=128, q_block_size=16, kv_block_size=384)
#test_flex_block_attn(batch_size=1, seq_len=256, head_num=1, head_dim=128, q_block_size=64, kv_block_size=128)
#test_flex_block_attn(batch_size=1, seq_len=256, head_num=1, head_dim=128, q_block_size=128, kv_block_size=64)
