import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
import argparse
import torch
import copy
from triton.testing import do_bench
from spas_sage_attn import block_sparse_sage2_attn_cuda
from utils.utils import (
    append_to_csv,
    generate_input,
    generate_selected_blocks,
    generate_swa_selected_blocks,
    create_block_mask,
    create_torch_mask,
    create_torch_swa_mask,
    torch_attn,
    create_flex_block_mask,
    create_flex_swa_mask,
    create_sparge_swa_mask,
    create_sparge_random_mask,
    torch_flex_attn,
    magi_get_qkranges,
    magi_swa_getqkranges,
    magi_attn,
    flash_attn,
    flash_attn3,
    ptm_sparse_attn,
    create_sparse_mask,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=71680)
    parser.add_argument("--num-heads", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--sparse-rate", type=float, default=0.2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--save", choices=["true", "false"], default="true")
    args = parser.parse_args()

    batch_size = args.batch_size
    seq_len = args.seq_len
    num_heads = args.num_heads
    head_dim = args.head_dim
    block_size = args.block_size

    device = torch.device("cuda:0")
    # with torch.no_grad():
    query, key, value = generate_input(batch_size, seq_len, num_heads, head_dim, device)

    # block_size = 384
    q = query.clone().transpose(1, 2).contiguous()
    k = key.clone().transpose(1, 2).contiguous()
    v = value.clone().transpose(1, 2).contiguous()

    selected_blocks = generate_selected_blocks(q, block_size, 1)
    torch_mask = None
    sparse_mask = create_sparse_mask(q, block_size, selected_blocks)
    flex_mask = None

    flex_attn_result = torch_flex_attn(q, k, v, flex_mask)
    flex_attn_result = flex_attn_result.transpose(1, 2).contiguous()
    flex_attn_result_sum = flex_attn_result.sum() * flex_attn_result.sum()
    torch_flex_bwd_time = do_bench(
        fn=lambda: flex_attn_result_sum.backward(retain_graph=True),
        warmup=2,
        rep=8,
        return_mode="mean",
    )
    flex_query_grad = query.grad
    query.grad.zero_()

    torch_attn_result = torch_attn(q, k, v, torch_mask)
    torch_attn_result = torch_attn_result.transpose(1, 2).contiguous()
    torch_attn_result_sum = torch_attn_result.sum() * torch_attn_result.sum()
    torch_attn_bwd_time = do_bench(
        fn=lambda: torch_attn_result_sum.backward(retain_graph=True),
        warmup=2,
        rep=8,
        return_mode="mean",
    )
    torch_query_grad = query.grad
    query.grad.zero_()

    # q_ranges,k_ranges,max_seqlen_q,max_seqlen_k,attn_type_map = magi_get_qkranges(query,key,value,block_size,selected_blocks)
    # magi_attn_result = magi_attn(query,key,value,q_ranges,k_ranges,max_seqlen_q,max_seqlen_k,attn_type_map)
    q_ranges = torch.tensor([[0, seq_len]], device=device, dtype=torch.int32)
    k_ranges = torch.tensor([[0, seq_len]], device=device, dtype=torch.int32)
    attn_type_map = torch.tensor([0], device=device, dtype=torch.int32)
    magi_attn_result = magi_attn(query, key, value, q_ranges, k_ranges, attn_type_map)
    magi_attn_result = magi_attn_result.view(batch_size, seq_len, num_heads, head_dim)
    magi_attn_result_sum = magi_attn_result.sum() * magi_attn_result.sum()
    magi_attn_bwd_time = do_bench(
        fn=lambda: magi_attn_result_sum.backward(retain_graph=True),
        warmup=2,
        rep=8,
        return_mode="mean",
    )
    magi_query_grad = query.grad
    query.grad.zero_()

    ptm_attn_result = ptm_sparse_attn(q, k, v, block_size, sparse_mask)
    ptm_attn_result = ptm_attn_result.transpose(1, 2).contiguous()
    ptm_attn_result_sum = ptm_attn_result.sum() * ptm_attn_result.sum()
    ptm_attn_bwd_time = do_bench(
        fn=lambda: ptm_attn_result_sum.backward(retain_graph=True),
        warmup=2,
        rep=8,
        return_mode="mean",
    )
    ptm_query_grad = query.grad
    query.grad.zero_()

    fa2_result = flash_attn(query, key, value)
    fa2_result_sum = fa2_result.sum() * fa2_result.sum()
    fa2_bwd_time = do_bench(
        fn=lambda: fa2_result_sum.backward(retain_graph=True),
        warmup=2,
        rep=8,
        return_mode="mean",
    )
    fa2_query_grad = query.grad
    query.grad.zero_()

    fa3_result = flash_attn3(query, key, value)
    fa3_result_sum = fa3_result.sum() * fa3_result.sum()
    fa3_bwd_time = do_bench(
        fn=lambda: fa3_result_sum.backward(retain_graph=True),
        warmup=2,
        rep=8,
        return_mode="mean",
    )
    fa3_query_grad = query.grad
    query.grad.zero_()

    # mask_id = torch.ones((batch_size,num_heads,seq_len//64,seq_len//64),requires_grad=True)
    # mask_id = mask_id.to(device)
    # sparge_attn_result = block_sparse_sage2_attn_cuda(q, k, v, mask_id=mask_id, scale=None, pvthreshd=40, attention_sink=False, tensor_layout="HND", return_sparsity=False)
    # print(f'sparge requires grad:{sparge_attn_result.requires_grad}')
    # sparge_attn_result_sum = sparge_attn_result.sum() * sparge_attn_result.sum()
    # sparge_bwd_time = do_bench(fn=lambda:sparge_attn_result_sum.backward(retain_graph=True),warmup=2,rep=8,return_mode="mean")
    # sparge_query_grad = query.grad
    # query.grad.zero_()

    # 有用
    print(f"flex_query_grad.shape:{flex_query_grad.shape}")
    print(f"torch_query_grad.shape:{torch_query_grad.shape}")
    print(f"magi_query_grad.shape:{magi_query_grad.shape}")
    print(f"ptm_query_grad.shape:{ptm_query_grad.shape}")
    print(f"fa2_query_grad.shape:{fa2_query_grad.shape}")
    print(f"fa3_query_grad.shape:{fa3_query_grad.shape}")
    # print(f'sparge_query_grad.shape:{sparge_query_grad.shape}')
    # print(f'sparge_query_grad:{sparge_query_grad}')

    # 有用
    result_torch_magi = torch.allclose(
        torch_query_grad, magi_query_grad, rtol=5e-3, atol=5e-3
    )
    result_ptm_torch = torch.allclose(
        ptm_query_grad, torch_query_grad, rtol=5e-3, atol=5e-3
    )
    result_torch_torch = torch.allclose(
        torch_query_grad, torch_query_grad, rtol=5e-3, atol=5e-3
    )
    result_torch_flex = torch.allclose(
        torch_query_grad, flex_query_grad, rtol=5e-3, atol=5e-3
    )
    result_torch_flash2 = torch.allclose(
        torch_query_grad, fa2_query_grad, rtol=5e-3, atol=5e-3
    )
    result_torch_flash3 = torch.allclose(
        torch_query_grad, fa3_query_grad, rtol=5e-3, atol=5e-3
    )
    # result_torch_sparge = torch.allclose(torch_query_grad,sparge_query_grad,rtol=5e-3,atol=5e-3)

    # 有用
    print(f"result_torch_magi:{result_torch_magi}")
    print(f"result_ptm_torch:{result_ptm_torch}")
    print(f"result_torch_torch:{result_torch_torch}")
    print(f"result_torch_flex:{result_torch_flex}")
    print(f"result_torch_flash2:{result_torch_flash2}")
    print(f"result_torch_flash3:{result_torch_flash3}")
    # print(f'result_torch_sparge:{result_torch_sparge}')
    print(f"input.dtype:{query.dtype}")

    diff_ptm_torch = torch.max(torch.abs(torch_query_grad - ptm_query_grad))
    print(f"diff_ptm_torch:{diff_ptm_torch}")

    diff_magi_torch = torch.max(torch.abs(torch_query_grad - magi_query_grad))
    print(f"diff_magi_torch:{diff_magi_torch}")

    diff_flex_torch = torch.max(torch.abs(torch_query_grad - flex_query_grad))
    print(f"diff_flex_torch:{diff_flex_torch}")

    diff_flash2_torch = torch.max(torch.abs(torch_query_grad - fa2_query_grad))
    print(f"diff_flash2_torch:{diff_flash2_torch}")

    diff_flash3_torch = torch.max(torch.abs(torch_query_grad - fa3_query_grad))
    print(f"diff_flash3_torch:{diff_flash3_torch}")

    # diff_sparge_torch = torch.max(torch.abs(torch_query_grad-sparge_query_grad))
    # print(f'diff_sparge_torch:{diff_sparge_torch}')

    print(f"fa2_bwd_time:{fa2_bwd_time}")
    print(f"fa3_bwd_time:{fa3_bwd_time}")
    print(f"torch_attn_bwd_time:{torch_attn_bwd_time}")
    print(f"torch_flex_bwd_time:{torch_flex_bwd_time}")
    print(f"ptm_attn_bwd_time:{ptm_attn_bwd_time}")
    print(f"magi_attn_bwd_time:{magi_attn_bwd_time}")
    # print(f'sparge_bwd_time:{sparge_bwd_time}')

    # 示例数据
    experiment_data = {}
    experiment_data["batch_size"] = batch_size
    experiment_data["seq_len"] = seq_len
    experiment_data["num_heads"] = num_heads
    experiment_data["head_dim"] = head_dim

    experiment_data["block_size"] = block_size

    experiment_data["torch_flex_bwd"] = torch_flex_bwd_time

    experiment_data["torch_attn_bwd"] = torch_attn_bwd_time

    experiment_data["magi_attn_bwd"] = magi_attn_bwd_time

    experiment_data["ptm_attn_bwd"] = ptm_attn_bwd_time

    experiment_data["fa2_bwd"] = fa2_bwd_time

    experiment_data["fa3_bwd"] = fa3_bwd_time

    # experiment_data['sparge_bwd'] = sparge_bwd_time

    experiment_data["result_torch_magi"] = result_torch_magi
    experiment_data["diff_magi_torch"] = diff_magi_torch

    experiment_data["result_ptm_torch"] = result_ptm_torch
    experiment_data["diff_ptm_torch"] = diff_ptm_torch

    experiment_data["result_torch_flex"] = result_torch_flex
    experiment_data["diff_flex_torch"] = diff_flex_torch

    experiment_data["result_torch_flash2"] = result_torch_flash2
    experiment_data["diff_flash2_torch"] = diff_flash2_torch

    experiment_data["result_torch_flash3"] = result_torch_flash3
    experiment_data["diff_flash3_torch"] = diff_flash3_torch

    # experiment_data['result_torch_sparge'] = result_torch_sparge
    # experiment_data['diff_sparge_torch'] = diff_sparge_torch

    # 使用方法1
    if args.save == "true":
        append_to_csv(experiment_data, f"../results/full_bwd_results_{block_size}.csv")
        print(f"results have been recorded")

    del query, key, value
    del flex_attn_result, magi_attn_result, torch_attn_result, ptm_attn_result
    # del flex_attn_result
    torch.cuda.empty_cache()

    # torch.cuda.memory._dump_snapshot("sparse_memory_snapshot.pickle")


if __name__ == "__main__":
    main()
