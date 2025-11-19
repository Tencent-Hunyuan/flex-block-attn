#!/bin/bash

# 定义参数数组
batch_sizes=(1)
# seq_lens=(3072 6144 9216 12288)  # 1024*3=3072, 1024*6=6144等
# seq_lens=(10240 20480 30720 40960 51200 61440 71680 81920 92160)
# seq_lens=(9216 21504 30720 43008 52224 61440 73728 82944 92160)
seq_lens=(11520 19200 30720 38400 46080 53760 61440 69120 76800 84480)
num_heads=(1)
# sparse_rates=(0.2 0.4 0.6 0.8)
sparse_rates=(1.0)
block_size=384
# block_size=64

# 计数器
count=0

# 嵌套循环测试所有组合
for bs in "${batch_sizes[@]}"; do
  for seq in "${seq_lens[@]}"; do
    for nh in "${num_heads[@]}"; do
      for sr in "${sparse_rates[@]}"; do
        # 构造实验名称
        exp_name="bs${bs}_seq${seq}_nh${nh}_sr${sr}"
        
        echo "========================================"
        echo "Running experiment $count: $exp_name"
        echo "========================================"
        
        # 执行Python脚本
        python3 bench_full_2_fwd.py \
          --batch-size "$bs" \
          --seq-len "$seq" \
          --num-heads "$nh" \
          --sparse-rate "$sr" \
          --block-size "$block_size" 
        
        # 计数器递增
        ((count++))
      done
    done
  done
done

echo "All experiments completed! Total runs: $count"