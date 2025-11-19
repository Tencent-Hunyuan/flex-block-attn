import torch
from torch.nn.attention.flex_attention import flex_attention,create_block_mask
from torch.nn.attention.flex_attention import or_masks, and_masks
import torch.nn.functional as F
from magi_attention.api import flex_flash_attn_func
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
    flash_attn_func
)
from flash_attn_interface import flash_attn_func as flash_attn_func_v3
from flex_block_attn import flex_block_attn_func

from spas_sage_attn import block_sparse_sage2_attn_cuda

import numpy as np 
import os
import copy
import random
import sys
import time
import argparse
# os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import csv

from triton.testing import do_bench


flex_attention = torch.compile(flex_attention,dynamic=False)

def append_to_csv(data, filename):
    """
    将数据追加到CSV文件的最后一行
    
    参数:
        data: 要写入的数据(列表或字典)
        filename: CSV文件名
    """
    # 检查文件是否存在，决定是否写入表头
    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # 如果是字典数据
        if isinstance(data, dict):
            fieldnames = data.keys()
            dict_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                dict_writer.writeheader()  # 文件不存在时写入表头
            dict_writer.writerow(data)
            
        # 如果是列表数据
        elif isinstance(data, (list, tuple)):
            if not file_exists:
                # 可以在这里写入自定义表头，或者不写表头
                pass
            writer.writerow(data)

def generate_input(batch_size, seq_len, num_heads, head_dim, device):
    # 随机生成查询、键、值
    query = torch.randn((batch_size, seq_len, num_heads, head_dim),dtype=torch.bfloat16,device=device,requires_grad=True)
    key = torch.randn((batch_size, seq_len, num_heads, head_dim),dtype=torch.bfloat16,device=device,requires_grad=True)
    value = torch.randn((batch_size, seq_len, num_heads, head_dim),dtype=torch.bfloat16,device=device,requires_grad=True)

    return query,key,value

def generate_selected_blocks(Q, block_size, percentage):
    seq_len = Q.size(2)
    block_num = int(seq_len / block_size)
    true_blocks_row = int(block_num * percentage)
    selected_blocks = []
    cols = []
    for i in range(block_num):
        cols.append(i)

    for i in range(block_num):
        selected_cols = random.sample(cols,true_blocks_row)
        for item in selected_cols:
            selected_blocks.append((i,item))

    return selected_blocks

def generate_swa_selected_blocks(Q, block_size, window_size):
    block_num = Q.size(2)//block_size
    selected_blocks=[]
    for i in range(block_num):
        for j in range(i+1-window_size,i+window_size):
            if j>=0 and j<=block_num-1:
                selected_blocks.append((i,j))
    return selected_blocks

def create_sparse_mask(Q, block_size,selected_blocks):

    seq_len = Q.size(2)
    block_num = int(seq_len / block_size)
    block_mask = np.full((block_num, block_num), False, dtype=bool)
    for i, j in selected_blocks:
        block_mask[i, j] = True
    block_mask = torch.tensor(block_mask, dtype=torch.bool, device=Q.device)

    return block_mask

def create_torch_mask(Q,block_size,selected_blocks):

    seq_len = Q.size(2)
    mask = np.full((seq_len, seq_len), False, dtype=bool)

    for i, j in selected_blocks:
        x_start, x_end = i * block_size, (i + 1) * block_size
        y_start, y_end = j * block_size, (j + 1) * block_size
        
        # 使用切片操作一次性设置整个块
        mask[x_start:x_end, y_start:y_end] = True

    torch_mask = torch.tensor(mask, dtype=torch.bool, device=Q.device)

    return torch_mask

def create_torch_swa_mask(Q,block_size,window_size):
    b,n,s,d = Q.shape
    torch_swa_mask = np.full((s,s),False, dtype=bool)
    block_num = s//block_size
    for i in range(block_num):
        row_min = max(0,i+1-window_size)
        row_max = min(block_num-1,i+window_size-1)
        torch_swa_mask[i][row_min*block_size:row_max*block_size] = True
    torch_swa_mask = torch.tensor(torch_swa_mask,dtype=torch.bool,device=Q.device)
    return torch_swa_mask

def torch_attn(query,key,value,mask): 
    if mask!="all":
        output = F.scaled_dot_product_attention(query,key,value,is_causal=False,attn_mask=mask)
    else:
        output = F.scaled_dot_product_attention(query,key,value,is_causal=False)

    return output

def create_flex_block_mask(query,torch_blocks):
    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
    
    def full_mask(b,h,q_idx,kv_idx):
        # return torch.tensor(True, dtype=torch.bool,device=q_idx.device)
        casual_masks = q_idx>=kv_idx
        decasual_masks = q_idx<kv_idx
        return casual_masks & decasual_masks

    def generate_block_mask(b,h,q_idx,kv_idx):
        return torch_blocks[q_idx,kv_idx]

    
    b,n,s,d = query.shape
    block_mask = create_block_mask(generate_block_mask,b, n, s, s, device=query.device)
    return block_mask

def create_flex_swa_mask(query,block_size,window_size):
    # print(f'block_size:{block_size}')

    token_window_size = (window_size)*block_size
    # print(f'token_window_size:{token_window_size}')
    def sliding_window(b,h,q_idx,kv_idx):
        distance = torch.abs(q_idx//block_size - kv_idx//block_size)
        return distance < window_size

    b,n,s,d = query.shape
    swa_mask = create_block_mask(sliding_window,b,n,s,s,device=query.device,BLOCK_SIZE=block_size)
    return swa_mask

def create_sparge_swa_mask(query,block_size,sparge_block_size,window_size):
    b,n,s,d = query.shape
    sparge_mask_size = query.size(2)//sparge_block_size
    sparge_mask = np.full((sparge_mask_size,sparge_mask_size),0)

    sparse_mask_size = query.size(2)//block_size
    sparse_mask_contain_sparge_num = block_size//sparge_block_size
    assert sparse_mask_contain_sparge_num>=1
    for i in range(sparse_mask_size):
        row_min = max(0,i+1-window_size)
        row_max = min(sparse_mask_size,i+window_size-1)
        sparge_mask[i*sparse_mask_contain_sparge_num:(i+1)*sparse_mask_contain_sparge_num][row_min*sparse_mask_contain_sparge_num:row_max*sparse_mask_contain_sparge_num]=1
    
    sparge_mask = torch.tensor(sparge_mask,device = query.device)
    # print(f'sparge_mask.shape:{sparge_mask.shape}')
    # print(f'query.shape:{query.shape};sparge_mask_size:{sparge_mask_size}')
    sparge_mask = sparge_mask.repeat(b,n,1,1)
    return sparge_mask

def create_sparge_random_mask(query,block_size,sparge_block_size,selected_block):
    b,n,s,d = query.shape
    sparge_block_num = s//sparge_block_size
    sparge_mask = np.full((sparge_block_num,sparge_block_num),False,dtype=bool)
    sparse_mask_contain_sparge_num = block_size//sparge_block_size
    assert sparse_mask_contain_sparge_num>=1
    for item in selected_block:
        sparge_mask[item[0]*sparse_mask_contain_sparge_num:(item[0]+1)*sparse_mask_contain_sparge_num][item[1]*sparse_mask_contain_sparge_num:(item[1]+1)*sparse_mask_contain_sparge_num] = True

    sparge_mask = torch.tensor(sparge_mask,device = query.device)
    sparge_mask = sparge_mask.repeat(b,n,1,1)
    return sparge_mask


def torch_flex_attn(query,key,value,block_mask):
    softmax_scale = query.shape[-1] ** (-0.5)
    output = flex_attention(
        query=query,
        key=key,
        value=value,
        # score_mod=score_mod,
        scale=softmax_scale,
        block_mask=block_mask
    )
    return output

def magi_get_qkranges(query,key,value,block_size,selected_blocks):
    b,s,n,d = query.shape
    if selected_blocks != "all":
        selected_num= len(selected_blocks)
        q_ranges = np.full((selected_num,2),0)
        k_ranges = np.full((selected_num,2),0)
        attn_type_map = np.full((selected_num),0)
        for i in range(len(selected_blocks)):
            q_ranges[i][0]=selected_blocks[i][0]*block_size
            q_ranges[i][1]=(selected_blocks[i][0]+1)*block_size
            k_ranges[i][0]=selected_blocks[i][1]*block_size
            k_ranges[i][1]=(selected_blocks[i][1]+1)*block_size

        q_ranges = torch.tensor(q_ranges,device=query.device, dtype=torch.int32)
        k_ranges = torch.tensor(k_ranges,device=query.device, dtype=torch.int32)
        attn_type_map = torch.tensor(attn_type_map,device=query.device, dtype=torch.int32)
        # for i in range(b):
        #     for item in selected_blocks:
        #         selected_q.append(item[0]+i*block_num)
        #         selected_kv.append(item[1]+i*block_num)

        # q_ranges = [] 
        # kv_ranges = []
        # attn_type_map = []
        # for i in range(len(selected_q)):
        #     q_ranges.append([selected_q[i]*block_size,(selected_q[i]+1)*block_size]) 
        #     kv_ranges.append([selected_kv[i]*block_size,(selected_kv[i]+1)*block_size]) 
        #     attn_type_map.append(0)
        
    else:
        ranges = []
        for i in range(b):
            ranges.append([i*s,(i+1)*s])
        q_ranges = torch.tensor(ranges, device=query.device, dtype=torch.int32)
        k_ranges = torch.tensor(ranges, device=key.device, dtype=torch.int32)
        attn_type_map = torch.tensor([0]*b,device=query.device, dtype=torch.int32)

    return q_ranges,k_ranges,attn_type_map

def magi_swa_getqkranges(query,key,value,block_size,window_size):
    q_ranges = []
    k_ranges = []
    attn_type_map = []
    block_num = query.size(1)//block_size
    for i in range(block_num):
        q_ranges.append([i*block_size,(i+1)*block_size])
        k_ranges.append([max(0,i-window_size+1)*block_size,min(i+window_size,block_num)*block_size])
        attn_type_map.append(0)
    q_ranges = torch.tensor(q_ranges,device=query.device,dtype=torch.int32)
    k_ranges = torch.tensor(k_ranges,device=query.device,dtype=torch.int32)
    attn_type_map = torch.tensor(attn_type_map,device=query.device,dtype=torch.int32)
    # print(f'q_ranges:{q_ranges}')
    return q_ranges,k_ranges,attn_type_map

def magi_attn(query,key,value,q_ranges,k_ranges,attn_type_map):
    b,s,n,d = query.shape
    query = query.view(b*s,n,d)
    key = key.view(b*s,n,d)
    value = value.view(b*s,n,d)
    softmax_scale = query.shape[-1] ** (-0.5)
    # softmax_scale = None
    # print(f'softmax_scale:{softmax_scale}')
    # magi_output = flex_flash_attn_func(query,key,value,q_ranges,k_ranges,max_seqlen_q,max_seqlen_k,attn_type_map,softmax_scale=softmax_scale,disable_fwd_atomic_reduction=False)
    magi_output = flex_flash_attn_func(query,key,value,q_ranges,k_ranges,attn_type_map,softmax_scale=softmax_scale)

    return magi_output[0]

def flash_attn(query,key,value):

    result =  flash_attn_func(query,key,value,causal=False)
    return result

def flash_attn3(query,key,value):

    result =  flash_attn_func_v3(query,key,value,causal=False)

    return result[0]

def ptm_sparse_attn(query,key,value,block_size,block_mask):
    output = flex_block_attn_func(query, key, value, block_size, block_size, block_mask) 

    return output