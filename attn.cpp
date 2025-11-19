#include <torch/extension.h>
#include <ATen/ATen.h>

#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cuda_runtime.h>


#ifdef TK_COMPILE_ATTN
extern std::vector<torch::Tensor> attention_forward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v,
    int q_block_size, int kv_block_size, torch::Tensor load_block_mask,
    torch::Tensor compute_block_mask, bool use_small_block_mode
);
extern std::vector<torch::Tensor> attention_backward(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o,
    torch::Tensor l_vec, torch::Tensor og,
    int q_block_size, int kv_block_size, torch::Tensor load_block_mask,
    torch::Tensor compute_block_mask, bool use_small_block_mode
);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Flex Block Attention Kernels"; // optional module docstring

#ifdef TK_COMPILE_ATTN
    m.def("attn_forward",  torch::wrap_pybind_function(attention_forward), "Bidirectional forward MHA. Takes Q,K,V,O in (B,H,N,D) where D must be 128, and N must be a multiple of 64. Additionally writes out norm vector L of shape (B,H,N), used in backward pass.");
    m.def("attn_backward", torch::wrap_pybind_function(attention_backward), "Bidirectional backward MHA. Takes Q,K,V,O,Og,Qg,Kg,Vg in (B,H,N,D) where D must be 128, and N must be a multiple of 64. Additionally requres norm vec l_vec, and (TODO) d_vec memory.");
#endif    
}
