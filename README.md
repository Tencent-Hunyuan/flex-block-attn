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
output = flex_block_attn_func(query, key, value, block_size, block_size, block_mask) 
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
- tile_size can be any multiple of 64, with 384 recommended (as we have performed additional optimizations for this size)
- The current attention_mask only supports block-level masking. block_mask supports two shapes: [seq_len, seq_len] or [batch, head_num, seq_len, seq_len]

### 🙏 Acknowledgments

This project stands on the shoulders of the following amazing projects and resources. We extend our sincere gratitude to:

- **[ThunderKittens](https://github.com/HazyResearch/ThunderKittens)** 
- **[flash-attention](https://github.com/Dao-AILab/flash-attention)** 
- **[MagiAttention](https://github.com/SandAI-org/MagiAttention)** 
- **[flex attention](https://github.com/meta-pytorch/attention-gym)** 
- **[SpargeAttn](https://github.com/thu-ml/SpargeAttn)**
- **[Triton](https://github.com/triton-lang/triton)**
- **[MoBA](https://github.com/MoonshotAI/MoBA)**


We are grateful to the entire open-source community for their invaluable contributions.