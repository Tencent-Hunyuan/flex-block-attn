import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
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
    sparse_rate = args.sparse_rate

    device = torch.device("cuda:0")
    # with torch.no_grad():
    query, key, value = generate_input(batch_size, seq_len, num_heads, head_dim, device)

    # block_size = 384
    q = query.clone().transpose(1, 2).contiguous()
    k = key.clone().transpose(1, 2).contiguous()
    v = value.clone().transpose(1, 2).contiguous()

    selected_blocks = generate_selected_blocks(q, block_size, sparse_rate)
    torch_mask = create_torch_mask(q, block_size, selected_blocks)

    flex_mask = create_flex_block_mask(q, torch_mask)
    del selected_blocks, torch_mask
    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    flex_attn_result = torch_flex_attn(q, k, v, flex_mask)
    flex_fwd_memory = torch.cuda.max_memory_allocated()
    print(f"flex fwd memory:{flex_fwd_memory/1024**2} MB")
    flex_attn_result = flex_attn_result.transpose(1, 2).contiguous()
    flex_attn_result_sum = flex_attn_result.sum() * flex_attn_result.sum()
    torch.cuda.reset_peak_memory_stats()
    flex_attn_result_sum.backward(retain_graph=True)
    flex_bwd_memory = torch.cuda.max_memory_allocated()
    print(f"flex bwd memory:{flex_bwd_memory/1024**2} MB")
    query.grad.zero_()
    del flex_attn_result, flex_attn_result_sum
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    selected_blocks = generate_selected_blocks(q, block_size, sparse_rate)
    torch_mask = create_torch_mask(q, block_size, selected_blocks)
    del selected_blocks
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch_attn_result = torch_attn(q, k, v, torch_mask)
    torch_fwd_memory = torch.cuda.max_memory_allocated()
    print(f"torch fwd memory:{torch_fwd_memory/1024**2} MB")
    torch_attn_result = torch_attn_result.transpose(1, 2).contiguous()
    torch_attn_result_sum = torch_attn_result.sum() * torch_attn_result.sum()
    torch_attn_result_sum.backward(retain_graph=True)
    torch_bwd_memory = torch.cuda.max_memory_allocated()
    print(f"torch bwd memory:{torch_bwd_memory/1024**2} MB")
    query.grad.zero_()
    del torch_attn_result, torch_attn_result_sum
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    selected_blocks = generate_selected_blocks(q, block_size, sparse_rate)
    q_ranges, k_ranges, attn_type_map = magi_get_qkranges(
        query, key, value, block_size, selected_blocks
    )
    del selected_blocks
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    magi_attn_result = magi_attn(query, key, value, q_ranges, k_ranges, attn_type_map)
    magi_fwd_memory = torch.cuda.max_memory_allocated()
    print(f"magi fwd memory:{magi_fwd_memory/1024**2} MB")
    magi_attn_result = magi_attn_result.view(batch_size, seq_len, num_heads, head_dim)
    magi_attn_result_sum = magi_attn_result.sum() * magi_attn_result.sum()
    torch.cuda.reset_peak_memory_stats()
    magi_attn_result_sum.backward(retain_graph=True)
    magi_bwd_memory = torch.cuda.max_memory_allocated()
    print(f"magi bwd memory:{magi_bwd_memory/1024**2} MB")
    query.grad.zero_()
    del q_ranges, k_ranges, attn_type_map, magi_attn_result, magi_attn_result_sum
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    selected_blocks = generate_selected_blocks(q, block_size, sparse_rate)
    mask_id = create_sparge_random_mask(q, block_size, 64, selected_blocks)
    del selected_blocks
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    sparge_attn_result = block_sparse_sage2_attn_cuda(
        q,
        k,
        v,
        mask_id=mask_id,
        scale=None,
        pvthreshd=40,
        attention_sink=False,
        tensor_layout="HND",
        return_sparsity=False,
    )
    sparge_fwd_memory = torch.cuda.max_memory_allocated()
    print(f"sparge fwd memory:{sparge_fwd_memory/1024**2} MB")
    del mask_id, sparge_attn_result
    torch.cuda.empty_cache()

    selected_blocks = generate_selected_blocks(q, block_size, sparse_rate)
    sparse_mask = create_sparse_mask(q, block_size, selected_blocks)
    del selected_blocks
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    ptm_attn_result = ptm_sparse_attn(q, k, v, block_size, sparse_mask)
    ptm_fwd_memory = torch.cuda.max_memory_allocated()
    print(f"ptm fwd memory:{ptm_fwd_memory/1024**2} MB")
    ptm_attn_result = ptm_attn_result.transpose(1, 2).contiguous()
    ptm_attn_result_sum = ptm_attn_result.sum() * ptm_attn_result.sum()
    torch.cuda.reset_peak_memory_stats()
    ptm_attn_result_sum.backward(retain_graph=True)
    ptm_bwd_memory = torch.cuda.max_memory_allocated()
    print(f"ptm bwd memory:{ptm_bwd_memory/1024**2} MB")
    query.grad.zero_()
    del sparse_mask, ptm_attn_result, ptm_attn_result_sum
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # 示例数据
    experiment_data = {}
    experiment_data["batch_size"] = batch_size
    experiment_data["seq_len"] = seq_len
    experiment_data["num_heads"] = num_heads
    experiment_data["head_dim"] = head_dim

    experiment_data["block_size"] = block_size
    experiment_data["sparse_rate"] = sparse_rate

    experiment_data["torch_fwd_memory"] = torch_fwd_memory
    experiment_data["torch_bwd_memory"] = torch_bwd_memory
    experiment_data["flex_fwd_memory"] = flex_fwd_memory
    experiment_data["flex_bwd_memory"] = flex_bwd_memory
    experiment_data["magi_fwd_memory"] = magi_fwd_memory
    experiment_data["magi_bwd_memory"] = magi_bwd_memory
    experiment_data["ptm_fwd_memory"] = ptm_fwd_memory
    experiment_data["ptm_bwd_memory"] = ptm_bwd_memory
    experiment_data["sparge_fwd_memory"] = sparge_fwd_memory

    # 使用方法1
    if args.save == "true":
        append_to_csv(
            experiment_data, f"../results/random_memory_results_{block_size}.csv"
        )
        print(f"results have been recorded")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
