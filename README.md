### Install Flex Block Attention
 ```bash
 git submodule update --init --recursive
 python setup.py install
 ```

### Usage
#### custom kernel
```python
from flex_block_attn import flex_block_attn_func
from utils.utils import create_sparse_mask
# create block mask with selected blocks
sparse_mask = create_sparse_mask(q, block_size, selected_blocks)
#compute 
output = flex_block_attn_func(query, key, value, q_block_size, k_block_size, block_mask) 
```
#### SSTA kernel
```python
from flex_block_attn import ssta_3d_attention
#thw: latent shape
hidden_states = ssta_3d_attention(query, key, value, thw,
                    topk=128,
                    tile_thw=(4,4,4),
                    kernel_thw=(3,3,3),
                    text_len=256,
                    sparse_type='ssta',
                    threshold=0.95,
                    pad_type='zero',
                    text_mask=text_mask).permute(0,2,1,3)
```
### Performance 
We provide performance comparisons in the `benchmark` folder, including measurements for mask creation time, forward/backward execution time, and GPU memory usage across the following attention types: full attention, sparse static attention, and sparse dynamic attention.
### Notes
- The current dim must be 128
- q tile_size can be any multiple of 64, k/v tile_size can be any multiple of 64, with 384 recommended (as we have performed additional optimizations for this size)
- The current attention_mask only supports block-level masking. block_mask supports two shapes: [seq_len, seq_len] or [batch, head_num, seq_len, seq_len]

### 🙏 Acknowledgments

This project stands on the shoulders of the following amazing projects and resources. We extend our sincere gratitude to:

- **[ThunderKittens](https://github.com/HazyResearch/ThunderKittens)** :Our project extends its computational engine, building additional logic layers while leveraging its core calculation capabilities. The underlying computational power is entirely provided by its excellent infrastructure.
- **[flash-attention](https://github.com/Dao-AILab/flash-attention)** :As the current standard for efficient attention computation in long-sequence scenarios, FlashAttention's high-performance computational design has provided us with substantial inspiration. It has served as a critical benchmark for validating both the performance and correctness of our own implementations.
- **[MagiAttention](https://github.com/SandAI-org/MagiAttention)** :As a purpose-built solution optimized for long-sequence scenarios in text-to-video generation, MagiAttention demonstrates excellent flexibility and performance through its development based on FA3. Being one of the most prominent projects in this domain, it has served as an essential reference for our performance and correctness validation.
- **[flex attention](https://github.com/meta-pytorch/attention-gym)** :As the high-performance, highly flexible dynamic sparse attention framework introduced in PyTorch 2.3+, it aims to support various attention patterns (dense/sparse/block-sparse) through a unified API, eliminating the need for hand-written CUDA kernels. Its seamless integration with the PyTorch ecosystem has been instrumental in our rapid validation process.
- **[SpargeAttn](https://github.com/thu-ml/SpargeAttn)**:
- **[Triton](https://github.com/triton-lang/triton)**:
- **[MoBA](https://github.com/MoonshotAI/MoBA)**:
- **[STA(Sliding Tile Attention)](https://github.com/hao-ai-lab/FastVideo)**:


We are grateful to the entire open-source community for their invaluable contributions.