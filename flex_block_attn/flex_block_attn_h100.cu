// # Define TORCH_COMPILE macro

#include "kittens.cuh"
#include <cooperative_groups.h>
#include <iostream>

constexpr int CONSUMER_WARPGROUPS = (3);
constexpr int PRODUCER_WARPGROUPS = (1);
constexpr int NUM_WARPGROUPS      = (CONSUMER_WARPGROUPS+PRODUCER_WARPGROUPS);
constexpr int NUM_WORKERS         = (NUM_WARPGROUPS*kittens::WARPGROUP_WARPS);

using namespace kittens;
namespace cg = cooperative_groups;

template<int D, int T> struct fwd_attend_ker_tile_dims {};
template<> struct fwd_attend_ker_tile_dims<128, 1> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (1*4*16);
    constexpr static int stages     = (2);
};
template<> struct fwd_attend_ker_tile_dims<128, 2> {
    constexpr static int tile_width = (128);
    constexpr static int qo_height  = (4*16);
    constexpr static int kv_height  = (2*4*16);
    constexpr static int stages     = (2);
};

template<int D, int T> struct fwd_globals {
    using q_tile    =         st_bf<fwd_attend_ker_tile_dims<D, T>::qo_height, fwd_attend_ker_tile_dims<D, T>::tile_width>;
    using k_tile    =         st_bf<fwd_attend_ker_tile_dims<D, T>::kv_height, fwd_attend_ker_tile_dims<D, T>::tile_width>;
    using v_tile    =         st_bf<fwd_attend_ker_tile_dims<D, T>::kv_height, fwd_attend_ker_tile_dims<D, T>::tile_width>;
    using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<D, T>::qo_height, fwd_attend_ker_tile_dims<D, T>::tile_width>>;
    using o_tile    =         st_bf<fwd_attend_ker_tile_dims<D, T>::qo_height, fwd_attend_ker_tile_dims<D, T>::tile_width>;

    using q_gl = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl = gl<bf16,  -1, -1, -1, -1, v_tile>;
    using l_gl = gl<float, -1, -1, -1, -1, l_col_vec>;
    using o_gl = gl<bf16,  -1, -1, -1, -1, o_tile>;

    q_gl q;
    k_gl k;
    v_gl v;
    l_gl l;
    o_gl o;

    const int N;
    const int hr;
};

template<int D, int T, int num_consumer, int num_warpgoups, int num_workers>
__global__  __launch_bounds__((NUM_WORKERS)*kittens::WARP_THREADS, 1)
void fwd_attend_ker(const __grid_constant__ fwd_globals<D, T> g,
                    const int ratio,
                    const bool* block_mask,
                    const int block_mask_dim,
                    const bool* small_block_mask,
                    const bool use_small_block_mode,
                    const int small_block_ratio) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    int warpid = kittens::warpid(), warpgroupid = warpid/kittens::WARPGROUP_WARPS;

    using K = fwd_attend_ker_tile_dims<D, T>;

    using q_tile    =         st_bf<K::qo_height, K::tile_width>;
    using k_tile    =         st_bf<K::kv_height, K::tile_width>;
    using v_tile    =         st_bf<K::kv_height, K::tile_width>;
    using l_col_vec = col_vec<st_fl<K::qo_height, K::tile_width>>;
    using o_tile    =         st_bf<K::qo_height, K::tile_width>;

    q_tile    (&q_smem)[num_consumer] = al.allocate<q_tile, num_consumer>();
    k_tile    (&k_smem)[K::stages]    = al.allocate<k_tile, K::stages>();
    v_tile    (&v_smem)[K::stages]    = al.allocate<v_tile, K::stages>();
    l_col_vec (&l_smem)[num_consumer] = al.allocate<l_col_vec, num_consumer>();
    auto      (*o_smem)               = reinterpret_cast<o_tile(*)>(q_smem);

    int kv_blocks   = g.N / (K::kv_height);
    int kv_head_idx = blockIdx.y / g.hr;
    int seq_idx     = blockIdx.x * num_consumer;

    int block_id = blockIdx.x;
    if (block_mask_dim == 4) block_id += (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x);
    block_id = block_id * num_consumer / ratio;

    int small_block_id = blockIdx.x * 4 * (num_warpgoups-1) + warpid;
    if (block_mask_dim == 4) small_block_id += 4 * (num_warpgoups-1) * (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x);
    small_block_id /= small_block_ratio;

    __shared__ kittens::semaphore qsmem_semaphore, k_smem_arrived[K::stages], v_smem_arrived[K::stages], compute_done[K::stages];
    if (threadIdx.x == 0) {
        init_semaphore(qsmem_semaphore, 0, 1);
        for (int j = 0; j < K::stages; j++) {
            init_semaphore(k_smem_arrived[j], 0, 1);
            init_semaphore(v_smem_arrived[j], 0, 1);
            init_semaphore(compute_done[j], num_consumer, 0);
        }

        tma::expect_bytes(qsmem_semaphore, sizeof(q_smem));

        for (int wg = 0; wg < num_consumer; wg++) {
            coord<q_tile> q_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + wg, 0};
            tma::load_async(q_smem[wg], g.q, q_tile_idx, qsmem_semaphore);
        }

        int count = 0;
        int j = 0;
        while(count < K::stages - 1 and j < kv_blocks) {
            bool mask = block_mask[(block_id * kv_blocks + j)*T/ratio];
            if (mask) {
                coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
                tma::expect_bytes(k_smem_arrived[count], sizeof(k_tile));
                tma::load_async(k_smem[count], g.k, kv_tile_idx, k_smem_arrived[count]);
                tma::expect_bytes(v_smem_arrived[count], sizeof(v_tile));
                tma::load_async(v_smem[count], g.v, kv_tile_idx, v_smem_arrived[count]);

                count++;
            }
            j++;
        }
        //TODO count must > 0
    }
    __syncthreads();

    int pipe_idx = K::stages - 1;

    if (warpgroupid == num_warpgoups-1) {
        warpgroup::decrease_registers<32>();

        if (warpid == num_workers-4) {
            int count = 0;
            for (auto j = 0; j < kv_blocks; j++) {
                bool mask = block_mask[(block_id * kv_blocks + j)*T/ratio];
                if (mask) {
                    if (count >= K::stages-1) {
                        coord<k_tile> kv_tile_idx = {blockIdx.z, kv_head_idx, j, 0};
                        tma::expect_bytes(k_smem_arrived[count%K::stages], sizeof(k_tile));
                        tma::load_async(k_smem[count%K::stages], g.k, kv_tile_idx, k_smem_arrived[(count)%K::stages]);
                        tma::expect_bytes(v_smem_arrived[count%K::stages], sizeof(v_tile));
                        tma::load_async(v_smem[count%K::stages], g.v, kv_tile_idx, v_smem_arrived[count%K::stages]);

                        wait(compute_done[(count - 1)%K::stages], ((count - 1)/K::stages)%2);
                    }

                    count++;
                }
            }
        }
    } else {
        warpgroup::increase_registers<160>();

        rt_fl<16, K::kv_height>  att_block;
        rt_bf<16, K::kv_height>  att_block_mma;
        rt_fl<16, K::tile_width> o_reg;

        col_vec<rt_fl<16, K::kv_height>> max_vec, norm_vec, max_vec_last_scaled, max_vec_scaled;

        neg_infty(max_vec);
        zero(norm_vec);
        zero(o_reg);

        wait(qsmem_semaphore, 0);

        int count = 0;
        if (use_small_block_mode) {
            bool flag = false;

            for (auto j = 0; j < kv_blocks; j++) {
                int kv_block_id = (block_id * kv_blocks + j)*T/ratio;
                bool mask = block_mask[(block_id * kv_blocks + j)*T/ratio];
                if (mask) {
                    wait(k_smem_arrived[(count)%K::stages], (count/K::stages)%2);
                    warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(count)%K::stages]);

                    copy(max_vec_last_scaled, max_vec);
                    mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f);

                    warpgroup::mma_async_wait();

                    int kv_small_block_id = ((small_block_id * kv_blocks + j)*T/ratio);
                    bool small_mask = small_block_mask[((small_block_id * kv_blocks + j)*T/ratio)];

                   if (flag) {
                        if (not small_mask) {
                            #pragma unroll
                            for (auto j = 0; j < (K::kv_height/kittens::TILE_ROW_DIM<bf16>); j++) {
                                auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(att_block.tiles[0][j]);
                                neg_infty  (attn_subtile);
                                __syncwarp();
                            }
                        }
                    } else if (small_mask) {
                       flag = true;
                    }

                    if (flag) {
                        //m1
                        row_max(max_vec, att_block, max_vec);
                        //x1 * scale
                        mul(att_block, att_block,    1.44269504089f*0.08838834764f);
                        //m1 * scale
                        mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
                        // x1 - m1
                        sub_row(att_block, att_block, max_vec_scaled);
                        // e(x1 - m1)
                        exp2(att_block, att_block);
                        // d0 - d1
                        sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                        // e(d0 - d1)
                        exp2(max_vec_last_scaled,       max_vec_last_scaled);
                        // norm_vec * m0
                        mul(norm_vec,            norm_vec,     max_vec_last_scaled);
                        // d1
                        row_sum(norm_vec,  att_block, norm_vec);
                        //?
                        add(att_block, att_block, 0.f);
                        //copy
                        copy(att_block_mma, att_block);
                        //o1*d0
                        mul_row(o_reg, o_reg, max_vec_last_scaled);
                    }

                    wait(v_smem_arrived[(count)%K::stages], (count/K::stages)%2);
                    //e(x1 - m1) * v1
                    warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(count)%K::stages]);
                    warpgroup::mma_async_wait();

                    if (not flag) {
                        zero(o_reg);
                    }

                    if(warpgroup::laneid() == 0) {
                        arrive(compute_done[(count)%K::stages], 1);
                    }
                    count++;
                }
            }

            if (flag) {
                //e(x1 - m1) * v1 / d1
                div_row(o_reg, o_reg, norm_vec);
            }
        } else {
            for (auto j = 0; j < kv_blocks; j++) {
                int kv_block_id = (block_id * kv_blocks + j)*T/ratio;
                bool mask = block_mask[(block_id * kv_blocks + j)*T/ratio];
                if (mask) {
                    wait(k_smem_arrived[(count)%K::stages], (count/K::stages)%2);
                    warpgroup::mm_ABt(att_block, q_smem[warpgroupid], k_smem[(count)%K::stages]);

                    copy(max_vec_last_scaled, max_vec);
                    mul(max_vec_last_scaled, max_vec_last_scaled, 1.44269504089f*0.08838834764f);

                    warpgroup::mma_async_wait();

                    //m1
                    row_max(max_vec, att_block, max_vec);
                    //x1 * scale
                    mul(att_block, att_block,    1.44269504089f*0.08838834764f);
                    //m1 * scale
                    mul(max_vec_scaled, max_vec, 1.44269504089f*0.08838834764f);
                    // x1 - m1
                    sub_row(att_block, att_block, max_vec_scaled);
                    // e(x1 - m1)
                    exp2(att_block, att_block);
                    // d0 - d1
                    sub(max_vec_last_scaled, max_vec_last_scaled, max_vec_scaled);
                    // e(d0 - d1)
                    exp2(max_vec_last_scaled,       max_vec_last_scaled);
                    // norm_vec * m0
                    mul(norm_vec,            norm_vec,     max_vec_last_scaled);
                    // d1
                    row_sum(norm_vec,  att_block, norm_vec);
                    //?
                    add(att_block, att_block, 0.f);
                    //copy
                    copy(att_block_mma, att_block);
                    //o1*d0
                    mul_row(o_reg, o_reg, max_vec_last_scaled);

                    wait(v_smem_arrived[(count)%K::stages], (count/K::stages)%2);
                    //e(x1 - m1) * v1
                    warpgroup::mma_AB(o_reg, att_block_mma, v_smem[(count)%K::stages]);
                    warpgroup::mma_async_wait();

                    if(warpgroup::laneid() == 0) arrive(compute_done[(count)%K::stages], 1);
                    count++;
                }
            }
            //e(x1 - m1) * v1 / d1
            div_row(o_reg, o_reg, norm_vec);
        }

        warpgroup::store(o_smem[warpgroupid], o_reg);
        warpgroup::sync(warpgroupid + num_warpgoups);

        if (warpid % num_warpgoups == 0) {
            coord<o_tile> o_tile_idx = {blockIdx.z, blockIdx.y, (seq_idx) + warpgroupid, 0};
            tma::store_async(g.o, o_smem[warpgroupid], o_tile_idx);
        }

        mul(max_vec_scaled,   max_vec_scaled, 0.69314718056f);
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec_scaled);

        mul(norm_vec, norm_vec, -11.313708499f);

        warpgroup::store(l_smem[warpgroupid], norm_vec);
        warpgroup::sync(warpgroupid + num_warpgoups);

        if (warpid % num_warpgoups == 0) {
            coord<l_col_vec> tile_idx = {blockIdx.z, blockIdx.y, 0, (seq_idx) + warpgroupid};
            tma::store_async(g.l, l_smem[warpgroupid], tile_idx);
        }
        tma::store_async_wait();
    }
}

// ---------------------------------------------------------------------------------------------------
// ----------------------------------- Backward preparation kernel -----------------------------------
// ---------------------------------------------------------------------------------------------------

template<int D>
struct bwd_prep_globals {
    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using o_gl  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    og_gl og;
    o_gl  o;
    d_gl  d;
};

template<int D, int W>
__global__  __launch_bounds__(4*kittens::WARP_THREADS, (D == 64) ? 2 : 1)
void bwd_attend_prep_ker(const __grid_constant__ bwd_prep_globals<D> g) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    int warpid = kittens::warpid();

    using og_tile = st_bf<4*16, D>;
    using o_tile  = st_bf<4*16, D>;
    using d_tile  = col_vec<st_fl<4*16, D>>;

    og_tile (&og_smem)[W] = al.allocate<og_tile, W>();
    o_tile  (&o_smem) [W] = al.allocate<o_tile , W>();
    d_tile  (&d_smem) [W] = al.allocate<d_tile , W>();

    rt_fl<4*16, D> og_reg, o_reg;
    col_vec<rt_fl<4*16, D>> d_reg;

    __shared__ kittens::semaphore smem_semaphore;

    if (threadIdx.x == 0) {
        init_semaphore(smem_semaphore, 0, 1);
        tma::expect_bytes(smem_semaphore, sizeof(og_smem[0]) * W * 2);
    }
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < W; w++) {
            coord<o_tile> tile_idx = {blockIdx.z, blockIdx.y, (blockIdx.x * W) + w, 0};
            tma::load_async(o_smem[w],  g.o,  tile_idx, smem_semaphore);
            tma::load_async(og_smem[w], g.og, tile_idx, smem_semaphore);
        }
    }

    wait(smem_semaphore, 0);
    load(o_reg, o_smem[warpid]);
    load(og_reg, og_smem[warpid]);
    mul(og_reg, og_reg, o_reg);
    row_sum(d_reg, og_reg);
    store(d_smem[warpid], d_reg);
    __syncthreads();

    if (warpid == 0) {
        for (int w = 0; w < W; w++) {
            coord<d_tile> tile_idx = {blockIdx.z, blockIdx.y, 0, (blockIdx.x * W) + w};
            tma::store_async(g.d, d_smem[w], tile_idx);
        }
    }
    tma::store_async_wait();
}

template<int D> struct bwd_attend_ker_tile_dims {};
template<> struct bwd_attend_ker_tile_dims<128> {
    constexpr static int tile_width = (128);
    constexpr static int tile_h     = (4*16);
    constexpr static int tile_h_qo  = (4*16);
    constexpr static int blocks_sm = 1;
};

constexpr int BWD_CONSUMER_WARPGROUPS = (2);
constexpr int BWD_PRODUCER_WARPGROUPS = (1);
constexpr int BWD_NUM_WARPGROUPS      = (BWD_CONSUMER_WARPGROUPS+BWD_PRODUCER_WARPGROUPS);
constexpr int BWD_NUM_WORKERS         = (BWD_NUM_WARPGROUPS*kittens::WARPGROUP_WARPS);

template<int D>
struct bwd_globals {
    using G = bwd_attend_ker_tile_dims<D>;

    using q_tile  =         st_bf<G::tile_h_qo, G::tile_width>;
    using k_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using v_tile  =         st_bf<G::tile_h,    G::tile_width>;
    using og_tile =         st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile =         st_fl<G::tile_h_qo, G::tile_width>;
    using kg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using vg_tile =         st_fl<G::tile_h,    G::tile_width>;
    using l_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile  = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;

    using q_gl  = gl<bf16,  -1, -1, -1, -1, q_tile>;
    using k_gl  = gl<bf16,  -1, -1, -1, -1, k_tile>;
    using v_gl  = gl<bf16,  -1, -1, -1, -1, v_tile>;

    using og_gl = gl<bf16,  -1, -1, -1, -1, og_tile>;

    using qg_gl = gl<float, -1, -1, -1, -1, qg_tile>;
    using kg_gl = gl<float, -1, -1, -1, -1, kg_tile>;
    using vg_gl = gl<float, -1, -1, -1, -1, vg_tile>;

    using l_gl  = gl<float, -1, -1, -1, -1, l_tile>;
    using d_gl  = gl<float, -1, -1, -1, -1, d_tile>;

    q_gl  q;
    k_gl  k;
    v_gl  v;
    og_gl og;
    qg_gl qg;
    kg_gl kg;
    vg_gl vg;
    l_gl  l;
    d_gl  d;

    const int N;
    const int hr;
};

__device__ static inline void
stream_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(kittens::laneid()%4);
        reg_tile.tiles[0][i].data[0] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[1] = *(float2*)&smem_vec[tic][base_col + 0];
        reg_tile.tiles[0][i].data[2] = *(float2*)&smem_vec[tic][base_col + 8];
        reg_tile.tiles[0][i].data[3] = *(float2*)&smem_vec[tic][base_col + 8];
    }
}

__device__ static inline void
stream_sub_tile(auto &reg_tile, auto &smem_vec, int tic) {
    #pragma unroll
    for(int i = 0; i < 4; i++) {
        int base_col = 16*i + 2*(laneid()%4);
        reg_tile.tiles[0][i].data[0] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[0], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[1] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[1], *(float2*)&smem_vec[tic][base_col + 0]);
        reg_tile.tiles[0][i].data[2] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[2], *(float2*)&smem_vec[tic][base_col + 8]);
        reg_tile.tiles[0][i].data[3] = base_ops::sub::template op<float2>(reg_tile.tiles[0][i].data[3], *(float2*)&smem_vec[tic][base_col + 8]);
    }
}

template<int tile_h_qo, int tile_h>
__device__ static inline void
causal_mask(auto &reg_tile, int qo_idx) {
    int q_blk = (qo_idx) * (tile_h_qo/kittens::TILE_ROW_DIM<bf16>);
    int k_blk = (blockIdx.x * BWD_CONSUMER_WARPGROUPS * (tile_h/kittens::TILE_ROW_DIM<bf16>))
                + ((kittens::warpid()/kittens::WARPGROUP_WARPS) * (tile_h/kittens::TILE_ROW_DIM<bf16>))
                + (kittens::warpid() % kittens::WARPGROUP_WARPS);

    for (int j = 0; j < (tile_h_qo/kittens::TILE_ROW_DIM<bf16>); j++) {
        int q_idx = q_blk + j;
        auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(reg_tile.tiles[0][j]);
        if      (q_idx  < k_blk) { neg_infty(attn_subtile); }
        else if (q_idx == k_blk) { make_causal_t(attn_subtile, attn_subtile, kittens::base_types::constants<float>::neg_infty()); }
    }
}

template<int tile_h_qo, int tile_h, int tile_width, int D, int num_consumer>
__device__ static inline void
compute_bwd_loop(
        kittens::semaphore *vec_b, kittens::semaphore *q_b, kittens::semaphore *o_b,
        rt_fl<16, 64> &s_block_t, rt_fl<16, 64> &dp_block_t,
        rt_fl<16, 64> &p_block_t, rt_fl<16, 64> &ds_block_t,
        rt_bf<16, 64> &p_block_t_mma,  rt_bf<16, 64> &ds_block_t_mma,
        rt_fl<16, tile_width> &kg_reg, rt_fl<16, tile_width> &vg_reg,
        auto &q_smem, auto &k_smem, auto &v_smem,
        auto &og_smem, auto &ds_smem, auto &l_smem, auto &d_smem,
        int qo_idx, int q_start, int tic, int toc, const bool* small_block_mask,
        const bool use_small_block_mode, const int small_block_ratio, int qo_blocks_tile_idx_base)
{
    wait(vec_b[tic], ((qo_idx - q_start)/2)%2);
    stream_tile(s_block_t, l_smem, tic);
    wait(q_b[tic], ((qo_idx - q_start)/2)%2);

    warpgroup::mma_ABt(s_block_t, k_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], q_smem[tic]);
    warpgroup::mma_commit_group();

    wait(o_b[tic], ((qo_idx - q_start)/2)%2);
    warpgroup::mm_ABt(dp_block_t, v_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], og_smem[tic]);
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    mul(s_block_t, s_block_t, 1.44269504089f*0.08838834764f);

    if (use_small_block_mode) {
        for (int j = 0; j < (tile_h_qo/kittens::TILE_ROW_DIM<bf16>); j++) {
            int qo_blocks_tile_idx = qo_blocks_tile_idx_base + j;
            qo_blocks_tile_idx /= small_block_ratio;

            if (small_block_mask[qo_blocks_tile_idx] == 0) {
                auto &attn_subtile = reinterpret_cast<rt_fl<16, 16>&>(s_block_t.tiles[0][j]);
                neg_infty(attn_subtile);
            }
        }
    }
    exp2(s_block_t, s_block_t);
    copy(p_block_t, s_block_t);
    copy(p_block_t_mma, s_block_t);
    stream_sub_tile(dp_block_t, d_smem, tic);
    mul(ds_block_t, p_block_t, dp_block_t);

    mul(ds_block_t, ds_block_t, 0.08838834764f);

    warpgroup::mma_AB(vg_reg, p_block_t_mma, og_smem[tic]);
    warpgroup::mma_commit_group();

    copy(ds_block_t_mma, ds_block_t);
    warpgroup::store(ds_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], ds_block_t);

    warpgroup::mma_AB(kg_reg, ds_block_t_mma, q_smem[tic]);
    warpgroup::mma_commit_group();
    warpgroup::mma_async_wait();

    group<4 * num_consumer>::sync(10);
}

template<typename kg_tile, typename vg_tile, int num_consumer>
__device__ static inline void
kv_store(auto &kg_smem, auto &kg_reg,
         auto &vg_smem, auto &vg_reg,
         auto &dst, auto &bar, int kv_head_idx, int toc)
{
    group<4 * num_consumer>::sync(10);
    warpgroup::store(kg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], kg_reg);

    group<4>::sync(warpgroup::groupid()+4);
    if (kittens::warpid() % 4 == 0) {
        coord<kg_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * num_consumer) + (kittens::warpid()/kittens::WARPGROUP_WARPS), 0};
        tma::store_add_async(dst.kg, kg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], tile_idx);
        tma::store_commit_group();
    }

    wait(bar, toc);
    warpgroup::store(vg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], vg_reg);
    group<4>::sync(warpgroup::groupid()+4);

    if (kittens::warpid() % 4 == 0) {
        coord<vg_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * num_consumer) + (kittens::warpid()/kittens::WARPGROUP_WARPS), 0};
        tma::store_add_async(dst.vg, vg_smem[kittens::warpid()/kittens::WARPGROUP_WARPS], tile_idx);
        tma::store_commit_group();
    }
    tma::store_async_wait();
}

template<int D, int num_consumer, int num_warpgoups, int num_workers>
__global__ __launch_bounds__(BWD_NUM_WORKERS*kittens::WARP_THREADS, bwd_attend_ker_tile_dims<D>::blocks_sm)
void bwd_attend_ker(const __grid_constant__ bwd_globals<D> g,
                    const int ratio,
                    const bool* block_mask,
                    const int block_mask_dim,
                    const bool* small_block_mask,
                    const bool use_small_block_mode,
                    const int small_block_ratio) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);

    const int N = g.N, hr = g.hr;
    using G = bwd_attend_ker_tile_dims<D>;

    using kg_tile   = st_fl<G::tile_h, G::tile_width>;
    using vg_tile   = st_fl<G::tile_h, G::tile_width>;
    using k_tile    = st_bf<G::tile_h, G::tile_width>;
    using v_tile    = st_bf<G::tile_h, G::tile_width>;
    using q_tile    = st_bf<G::tile_h_qo, G::tile_width>;
    using og_tile   = st_bf<G::tile_h_qo, G::tile_width>;
    using qg_tile   = st_fl<G::tile_h_qo, G::tile_width>;
    using l_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using d_tile    = row_vec<st_fl<G::tile_h_qo, G::tile_h>>;
    using attn_tile = st_bf<G::tile_h_qo, G::tile_h>;

    k_tile  (&k_smem) [num_consumer] = al.allocate<k_tile, num_consumer>();
    v_tile  (&v_smem) [num_consumer] = al.allocate<v_tile, num_consumer>();

    q_tile  (&q_smem) [2] = al.allocate<q_tile,  2>();
    og_tile (&og_smem)[2] = al.allocate<og_tile, 2>();
    qg_tile (&qg_smem)    = al.allocate<qg_tile>();

    l_tile   (&l_smem)[2] = al.allocate<l_tile, 2>();
    d_tile   (&d_smem)[2] = al.allocate<d_tile, 2>();
    kg_tile (*kg_smem)    = reinterpret_cast<kg_tile*>(&k_smem[0].data[0]);
    vg_tile (*vg_smem)    = reinterpret_cast<vg_tile*>(&q_smem[0].data[0]);

    attn_tile (&ds_smem)[num_consumer] = al.allocate<attn_tile, num_consumer>();

    const int warpid      = kittens::warpid();
    const int warpgroupid = warpid / kittens::WARPGROUP_WARPS;
    const int qo_blocks   = N / (G::tile_h_qo);
    const int kv_head_idx = (blockIdx.y) / hr;

    int block_id = blockIdx.x * num_consumer / ratio;
    if (block_mask_dim == 4) block_id += (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x) * num_consumer / ratio;

    __shared__ kittens::semaphore kv_b, q_b[2], o_b[2], vec_b[2];
    __shared__ kittens::semaphore compute_done[2], qg_ready;

    int tic = 0, toc = 1;

    if (threadIdx.x == 0) {
        init_semaphore(kv_b,  0, 1);
        init_semaphore(qg_ready, 1, 0);
        for (int s = 0; s < 2; s++) {
            init_semaphore(q_b[s],   0, 1);
            init_semaphore(o_b[s],   0, 1);
            init_semaphore(vec_b[s], 0, 1);
            init_semaphore(compute_done[s], 1, 0);
        }

        tma::expect_bytes(kv_b, (sizeof(k_smem[0]) + sizeof(v_smem[0])) * num_consumer);
        for (int w = 0; w < num_consumer; w++) {
            coord<k_tile> tile_idx = {blockIdx.z, kv_head_idx, (blockIdx.x * num_consumer) + w, 0};
            tma::load_async(k_smem[w], g.k, tile_idx, kv_b);
            tma::load_async(v_smem[w], g.v, tile_idx, kv_b);
        }

        int j = 0;
        int count = 0;
        while (count < 1 and j < qo_blocks) {
            bool mask = block_mask[(block_id * qo_blocks + j)/ratio];
            if (mask) {
                coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, j, 0};
                tma::expect_bytes(q_b[tic],   sizeof(q_smem[0]));
                tma::load_async(q_smem[tic],  g.q,  tile_idx, q_b[tic]);
                tma::expect_bytes(o_b[tic],   sizeof(og_smem[0]));
                tma::load_async(og_smem[tic], g.og, tile_idx, o_b[tic]);

                coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, j};
                tma::expect_bytes(vec_b[tic], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                tma::load_async(l_smem[tic], g.l, vec_idx, vec_b[tic]);
                tma::load_async(d_smem[tic], g.d, vec_idx, vec_b[tic]);

                count++;
            }
            j++;
        }
    }
    __syncthreads();

    if (warpgroupid == num_warpgoups - 1) {
        warpgroup::decrease_registers<24>();

        if (warpid % kittens::WARPGROUP_WARPS == 0) {
            int count = 0;
            bool flag = false;
            for (auto j = 0; j < qo_blocks; j++) {
                bool mask = block_mask[(block_id * qo_blocks + j)/ratio];
                if (mask) {
                    if (count > 0) {
                        coord<q_tile> tile_idx = {blockIdx.z, blockIdx.y, j, 0};
                        tma::expect_bytes(q_b[toc],   sizeof(q_smem[0]));
                        tma::load_async(q_smem[toc], g.q,  tile_idx, q_b[toc]);
                        tma::expect_bytes(o_b[toc],   sizeof(og_smem[0]));
                        tma::load_async(og_smem[toc], g.og, tile_idx, o_b[toc]);

                        coord<l_tile> vec_idx = {blockIdx.z, blockIdx.y, 0, j};
                        tma::expect_bytes(vec_b[toc], sizeof(l_smem[0]) + sizeof(d_smem[0]));
                        tma::load_async(l_smem[toc], g.l, vec_idx, vec_b[toc]);
                        tma::load_async(d_smem[toc], g.d, vec_idx, vec_b[toc]);

                        wait(compute_done[tic], ((count - 1)/(2))%2);

                        tic ^= 1;
                        toc ^= 1;
                    }

                    count++;
                    flag = true;
                }
            }
            if (flag) {
                wait(compute_done[tic], ((count - 1)/(2))%2);
            }
        }
        else if(warpid % kittens::WARPGROUP_WARPS == 1) {
            int count = 0;
            for (auto j = 0; j < qo_blocks; j++) {
                bool mask = block_mask[(block_id * qo_blocks + j)/ratio];
                if (mask) {
                    wait(compute_done[tic], ((count)/(2))%2);

                    coord<qg_tile> tile_idx = {blockIdx.z, blockIdx.y, j, 0};
                    tma::store_add_async(g.qg, qg_smem, tile_idx);
                    tma::store_async_wait();

                    if(laneid() == 0) arrive(qg_ready);

                    count++;
                    tic ^= 1;
                    toc ^= 1;
                }
            }
        }
    }
    else {
        rt_fl<16, G::tile_width> kg_reg, vg_reg;

        row_vec<rt_fl<16, 64>> row_reg;

        rt_fl<16, 64> s_block_t,  p_block_t;
        rt_fl<16, 64> ds_block_t, dp_block_t;
        rt_bf<16, 64> ds_block_t_mma, p_block_t_mma;

        zero(kg_reg);
        zero(vg_reg);

        if (warpgroupid == 0) {
            warpgroup::increase_registers<256>();
            wait(kv_b, 0);

            int count = 0;
            for (auto j = 0; j < qo_blocks; j++) {
                bool mask = block_mask[(block_id * qo_blocks + j)/ratio];
                if (mask) {
                    int qo_blocks_tile_idx_base = (block_id * qo_blocks + j) * 4;
                    compute_bwd_loop<G::tile_h_qo, G::tile_h, G::tile_width, D, num_consumer>(
                        vec_b, q_b, o_b,
                        s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                        kg_reg, vg_reg,
                        q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                        count, 0, tic, toc, small_block_mask, use_small_block_mode,
                        small_block_ratio, qo_blocks_tile_idx_base
                    );

                    rt_fl<16, G::tile_width> qg_reg;
                    warpgroup::mm_AtB(qg_reg, ds_smem[0], k_smem[0]);
                    if (num_consumer > 1) warpgroup::mma_AtB(qg_reg, ds_smem[1], k_smem[1]);
                    warpgroup::mma_commit_group();

                    wait(qg_ready, toc);

                    if (count > 0) tma::store_async_wait();

                    warpgroup::mma_async_wait();

                    warpgroup::store(qg_smem, qg_reg);
                    group<4>::sync(warpgroup::groupid()+4);

                    if (warpgroup::laneid() == 0) arrive(compute_done[tic]);

                    count++;
                    tic ^= 1;
                    toc ^= 1;
                }
            }

            kv_store<kg_tile, vg_tile, num_consumer>(kg_smem, kg_reg, vg_smem, vg_reg, g, qg_ready, kv_head_idx, toc);
        }
        else {
            warpgroup::increase_registers<224>();
            wait(kv_b, 0);

            int count = 0;
            for (auto j = 0; j < qo_blocks; j++) {
                bool mask = block_mask[(block_id * qo_blocks + j)/ratio];
                if (mask) {
                    int qo_blocks_tile_idx_base = (block_id * qo_blocks + j) * 4;

                    compute_bwd_loop<G::tile_h_qo, G::tile_h, G::tile_width, D, num_consumer>(
                        vec_b, q_b, o_b,
                        s_block_t, dp_block_t, p_block_t, ds_block_t, p_block_t_mma, ds_block_t_mma,
                        kg_reg, vg_reg,
                        q_smem, k_smem, v_smem, og_smem, ds_smem, l_smem, d_smem,
                        count, 0, tic, toc, small_block_mask, use_small_block_mode,
                        small_block_ratio, qo_blocks_tile_idx_base
                    );

                    count++;
                    tic ^= 1;
                    toc ^= 1;
                }
            }

            kv_store<kg_tile, vg_tile, num_consumer>(kg_smem, kg_reg, vg_smem, vg_reg, g, qg_ready, kv_head_idx, toc);
        }
    }
}

#ifdef TORCH_COMPILE

#include "pyutils/torch_helpers.cuh"
#include <ATen/cuda/CUDAContext.h>
#include <iostream>

std::vector<torch::Tensor>
attention_forward(torch::Tensor q,
                  torch::Tensor k,
                  torch::Tensor v,
                  const int q_block_size,
                  const int kv_block_size,
                  torch::Tensor block_mask,
                  torch::Tensor small_block_mask,
                  bool use_small_block_mode)
{
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto batch    = q.size(0);
    auto q_seq_len  = q.size(2);
    auto head_dim = q.size(3);
    auto qo_heads = q.size(1);

    auto kv_seq_len  = k.size(2);
    auto kv_heads = k.size(1);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(head_dim == 128, "head_dim must be 128");

    TORCH_CHECK(q.size(0) == batch, "Q batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0) == batch, "K batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0) == batch, "V batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(3) == head_dim, "Q head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3) == head_dim, "K head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3) == head_dim, "V head dimension - idx 3 - must match for all non-vector inputs");

    TORCH_CHECK(qo_heads >= kv_heads, "QO heads must be greater than or equal to KV heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "QO heads must be divisible by KV heads");
    TORCH_CHECK(q.size(1) == qo_heads, "QO head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1) == kv_heads, "KV head dimension - idx 1 - must match for all inputs");

    const int block_mask_dim = small_block_mask.dim();
    TORCH_CHECK(block_mask_dim == 2 or block_mask_dim == 4, "block_mask dim must be 2/4");

    auto hr = qo_heads / kv_heads;

    c10::BFloat16* q_ptr = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr = v.data_ptr<c10::BFloat16>();
    bool* block_mask_ptr = block_mask.data_ptr<bool>();
    bool* small_block_mask_ptr = small_block_mask.data_ptr<bool>();

    bf16*  d_q = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v = reinterpret_cast<bf16*>(v_ptr);

    // for the returned outputs
    torch::Tensor o     = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(q_seq_len),
                                        static_cast<const uint>(head_dim)}, v.options());

    torch::Tensor l_vec = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(q_seq_len),
                                        static_cast<const uint>(1)},
                                        torch::TensorOptions().dtype(torch::kFloat).device(q.device()).memory_format(at::MemoryFormat::Contiguous));

    bf16*  o_ptr = reinterpret_cast<bf16*>(o.data_ptr<c10::BFloat16>());
    bf16*  d_o   = reinterpret_cast<bf16*>(o_ptr);

    float* l_ptr = reinterpret_cast<float*>(l_vec.data_ptr<float>());
    float* d_l   = reinterpret_cast<float*>(l_ptr);

    cudaDeviceSynchronize();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto mem_size = kittens::MAX_SHARED_MEMORY;
    const int ratio = kv_block_size / 64;
    const int small_block_ratio = q_block_size / 16;

    if (kv_block_size % 128 == 0) {
        using q_tile    =         st_bf<fwd_attend_ker_tile_dims<128, 2>::qo_height, fwd_attend_ker_tile_dims<128, 2>::tile_width>;
        using k_tile    =         st_bf<fwd_attend_ker_tile_dims<128, 2>::kv_height, fwd_attend_ker_tile_dims<128, 2>::tile_width>;
        using v_tile    =         st_bf<fwd_attend_ker_tile_dims<128, 2>::kv_height, fwd_attend_ker_tile_dims<128, 2>::tile_width>;
        using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128, 2>::qo_height, fwd_attend_ker_tile_dims<128, 2>::tile_width>>;
        using o_tile    =         st_bf<fwd_attend_ker_tile_dims<128, 2>::qo_height, fwd_attend_ker_tile_dims<128, 2>::tile_width>;

        using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

        using globals      = fwd_globals<128, 2>;

        q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(q_seq_len), 128U};
        k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kv_seq_len), 128U};
        v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kv_seq_len), 128U};
        l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(q_seq_len)};
        o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(q_seq_len), 128U};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(kv_seq_len), static_cast<int>(hr)};

        if (kv_block_size % 384 == 0) {
            constexpr int num_consumer = 3;
            constexpr int num_warpgoups = num_consumer + PRODUCER_WARPGROUPS;
            constexpr int num_workers = num_warpgoups * kittens::WARPGROUP_WARPS;

            cudaFuncSetAttribute(
                fwd_attend_ker<128, 2, num_consumer, num_warpgoups, num_workers>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            auto threads = num_workers * kittens::WARP_THREADS;
            dim3 grid(q_seq_len/(num_consumer*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);

            fwd_attend_ker<128, 2, num_consumer, num_warpgoups, num_workers><<<grid, threads, mem_size, stream>>>(g, ratio, block_mask_ptr, block_mask_dim,
                                                                                                                  small_block_mask_ptr, use_small_block_mode,
                                                                                                                  small_block_ratio);
        } else if (kv_block_size % 256 == 0) {
            constexpr int num_consumer = 2;
            constexpr int num_warpgoups = num_consumer + PRODUCER_WARPGROUPS;
            constexpr int num_workers = num_warpgoups * kittens::WARPGROUP_WARPS;

            cudaFuncSetAttribute(
                fwd_attend_ker<128, 2, num_consumer, num_warpgoups, num_workers>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            auto threads = num_workers * kittens::WARP_THREADS;
            dim3 grid(q_seq_len/(num_consumer*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);

            fwd_attend_ker<128, 2, num_consumer, num_warpgoups, num_workers><<<grid, threads, mem_size, stream>>>(g, ratio, block_mask_ptr, block_mask_dim,
                                                                                                                  small_block_mask_ptr, use_small_block_mode,
                                                                                                                  small_block_ratio);
        } else if (kv_block_size % 128 == 0) {
            constexpr int num_consumer = 1;
            constexpr int num_warpgoups = num_consumer + PRODUCER_WARPGROUPS;
            constexpr int num_workers = num_warpgoups * kittens::WARPGROUP_WARPS;

            cudaFuncSetAttribute(
                fwd_attend_ker<128, 2, num_consumer, num_warpgoups, num_workers>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                mem_size
            );

            auto threads = num_workers * kittens::WARP_THREADS;
            dim3 grid(q_seq_len/(num_consumer*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);

            fwd_attend_ker<128, 2, num_consumer, num_warpgoups, num_workers><<<grid, threads, mem_size, stream>>>(g, ratio, block_mask_ptr, block_mask_dim,
                                                                                                                  small_block_mask_ptr, use_small_block_mode,
                                                                                                                  small_block_ratio);
        }
    } else if (kv_block_size % 64 == 0) {
        using q_tile    =         st_bf<fwd_attend_ker_tile_dims<128, 1>::qo_height, fwd_attend_ker_tile_dims<128, 1>::tile_width>;
        using k_tile    =         st_bf<fwd_attend_ker_tile_dims<128, 1>::kv_height, fwd_attend_ker_tile_dims<128, 1>::tile_width>;
        using v_tile    =         st_bf<fwd_attend_ker_tile_dims<128, 1>::kv_height, fwd_attend_ker_tile_dims<128, 1>::tile_width>;
        using l_col_vec = col_vec<st_fl<fwd_attend_ker_tile_dims<128, 1>::qo_height, fwd_attend_ker_tile_dims<128, 1>::tile_width>>;
        using o_tile    =         st_bf<fwd_attend_ker_tile_dims<128, 1>::qo_height, fwd_attend_ker_tile_dims<128, 1>::tile_width>;

        using q_global = gl<bf16,  -1, -1, -1, -1, q_tile>;
        using k_global = gl<bf16,  -1, -1, -1, -1, k_tile>;
        using v_global = gl<bf16,  -1, -1, -1, -1, v_tile>;
        using l_global = gl<float, -1, -1, -1, -1, l_col_vec>;
        using o_global = gl<bf16,  -1, -1, -1, -1, o_tile>;

        using globals      = fwd_globals<128, 1>;

        q_global qg_arg{d_q, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(q_seq_len), 128U};
        k_global kg_arg{d_k, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kv_seq_len), 128U};
        v_global vg_arg{d_v, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kv_seq_len), 128U};
        l_global lg_arg{d_l, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(q_seq_len)};
        o_global og_arg{d_o, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(q_seq_len), 128U};

        globals g{qg_arg, kg_arg, vg_arg, lg_arg, og_arg, static_cast<int>(kv_seq_len), static_cast<int>(hr)};

        constexpr int num_consumer = 1;
        constexpr int num_warpgoups = num_consumer + PRODUCER_WARPGROUPS;
        constexpr int num_workers = num_warpgoups * kittens::WARPGROUP_WARPS;
        cudaFuncSetAttribute(
            fwd_attend_ker<128, 1, num_consumer, num_warpgoups, num_workers>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        dim3 grid(q_seq_len/(num_consumer*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);
        auto threads = num_workers * kittens::WARP_THREADS;
        fwd_attend_ker<128, 1, num_consumer, num_warpgoups, num_workers><<<grid, threads, mem_size, stream>>>(g, ratio, block_mask_ptr, block_mask_dim,
                                                                                                              small_block_mask_ptr, use_small_block_mode,
                                                                                                              small_block_ratio);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();
    return {o, l_vec};
}

std::vector<torch::Tensor>
attention_backward(torch::Tensor q,
                   torch::Tensor k,
                   torch::Tensor v,
                   torch::Tensor o,
                   torch::Tensor l_vec,
                   torch::Tensor og,
                   const int q_block_size,
                   const int kv_block_size,
                   torch::Tensor block_mask,
                   torch::Tensor small_block_mask,
                   bool use_small_block_mode) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(l_vec);
    CHECK_INPUT(o);
    CHECK_INPUT(og);

    auto batch    = q.size(0);
    auto q_seq_len  = q.size(2);
    auto head_dim = q.size(3);

    auto kv_seq_len  = k.size(2);

    // check to see that these dimensions match for all inputs
    TORCH_CHECK(head_dim == 128, "head_dim must be 128");

    TORCH_CHECK(q.size(0)     == batch, "Q  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(k.size(0)     == batch, "K  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(v.size(0)     == batch, "V  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(l_vec.size(0) == batch, "L  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(o.size(0)     == batch, "O  batch dimension - idx 0 - must match for all inputs");
    TORCH_CHECK(og.size(0)    == batch, "OG batch dimension - idx 0 - must match for all inputs");

    TORCH_CHECK(q.size(3)  == head_dim, "Q  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(k.size(3)  == head_dim, "K  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(v.size(3)  == head_dim, "V  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(o.size(3)  == head_dim, "O  head dimension - idx 3 - must match for all non-vector inputs");
    TORCH_CHECK(og.size(3) == head_dim, "OG head dimension - idx 3 - must match for all non-vector inputs");

    auto qo_heads = q.size(1);
    auto kv_heads = k.size(1);

    TORCH_CHECK(qo_heads >= kv_heads,     "Q heads must be greater than or equal to K and V heads");
    TORCH_CHECK(qo_heads % kv_heads == 0, "Q heads must be divisible by KV heads");

    TORCH_CHECK(q.size(1)     == qo_heads, "Q  heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(l_vec.size(1) == qo_heads, "L  heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(o.size(1)     == qo_heads, "O  heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(og.size(1)    == qo_heads, "OG heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(k.size(1)  == kv_heads, "K  heads dimension - idx 1 - must match for all inputs");
    TORCH_CHECK(v.size(1)  == kv_heads, "V  heads dimension - idx 1 - must match for all inputs");

    auto hr = qo_heads / kv_heads;

    const int block_mask_dim = block_mask.dim();
    TORCH_CHECK(block_mask_dim == 2 or block_mask_dim == 4, "block_mask dim must be 2/4");

    c10::BFloat16* q_ptr  = q.data_ptr<c10::BFloat16>();
    c10::BFloat16* k_ptr  = k.data_ptr<c10::BFloat16>();
    c10::BFloat16* v_ptr  = v.data_ptr<c10::BFloat16>();
    c10::BFloat16* o_ptr  = o.data_ptr<c10::BFloat16>();
    c10::BFloat16* og_ptr = og.data_ptr<c10::BFloat16>();
    float*         l_ptr  = l_vec.data_ptr<float>();
    bool* block_mask_ptr = block_mask.data_ptr<bool>();
    bool* small_block_mask_ptr = small_block_mask.data_ptr<bool>();

    torch::Tensor qg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(qo_heads),
                                     static_cast<const uint>(q_seq_len),
                                     static_cast<const uint>(head_dim)},   l_vec.options());
    torch::Tensor kg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(kv_heads),
                                     static_cast<const uint>(kv_seq_len),
                                     static_cast<const uint>(head_dim)},   l_vec.options());
    torch::Tensor vg = torch::zeros({static_cast<const uint>(batch),
                                     static_cast<const uint>(kv_heads),
                                     static_cast<const uint>(kv_seq_len),
                                     static_cast<const uint>(head_dim)},   l_vec.options());

    torch::Tensor d_vec = torch::empty({static_cast<const uint>(batch),
                                        static_cast<const uint>(qo_heads),
                                        static_cast<const uint>(q_seq_len),
                                        static_cast<const uint>(1)},       l_vec.options());

    float*         qg_ptr = qg.data_ptr<float>();
    float*         kg_ptr = kg.data_ptr<float>();
    float*         vg_ptr = vg.data_ptr<float>();
    float*         d_ptr  = d_vec.data_ptr<float>();

    bf16*  d_q  = reinterpret_cast<bf16*>(q_ptr);
    bf16*  d_k  = reinterpret_cast<bf16*>(k_ptr);
    bf16*  d_v  = reinterpret_cast<bf16*>(v_ptr);
    bf16*  d_o  = reinterpret_cast<bf16*>(o_ptr);
    bf16*  d_og = reinterpret_cast<bf16*>(og_ptr);
    float* d_l  = reinterpret_cast<float*>(l_ptr);
    float* d_d  = reinterpret_cast<float*>(d_ptr);
    float* d_qg = reinterpret_cast<float*>(qg_ptr);
    float* d_kg = reinterpret_cast<float*>(kg_ptr);
    float* d_vg = reinterpret_cast<float*>(vg_ptr);

    auto mem_size = kittens::MAX_SHARED_MEMORY;

    cudaDeviceSynchronize();
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    using og_tile = st_bf<4*16, 128>;
    using o_tile  = st_bf<4*16, 128>;
    using d_tile  = col_vec<st_fl<4*16, 128>>;

    using og_global = gl<bf16,  -1, -1, -1, -1, og_tile>;
    using o_global  = gl<bf16,  -1, -1, -1, -1, o_tile>;
    using d_global  = gl<float, -1, -1, -1, -1, d_tile>;

    using bwd_prep_globals = bwd_prep_globals<128>;

    og_global prep_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(q_seq_len), 128U};
    o_global  prep_o_arg {d_o,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(q_seq_len), 128U};
    d_global  prep_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(q_seq_len)};

    bwd_prep_globals bwd_g{prep_og_arg, prep_o_arg, prep_d_arg};

    if (kv_seq_len % 256 == 0) {
        cudaFuncSetAttribute(
            bwd_attend_prep_ker<128, 4>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        auto threads  = 4 * kittens::WARP_THREADS;
        dim3 grid_bwd(kv_seq_len/(4*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);
        bwd_attend_prep_ker<128, 4><<<grid_bwd, threads, mem_size, stream>>>(bwd_g);
    } else if (kv_seq_len % 128 == 0) {
        cudaFuncSetAttribute(
            bwd_attend_prep_ker<128, 2>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        auto threads  = 2 * kittens::WARP_THREADS;
        dim3 grid_bwd(kv_seq_len/(2*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);
        bwd_attend_prep_ker<128, 2><<<grid_bwd, threads, mem_size, stream>>>(bwd_g);
    } else if (kv_seq_len % 64 == 0) {
        cudaFuncSetAttribute(
            bwd_attend_prep_ker<128, 1>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            mem_size
        );

        auto threads  = 1 * kittens::WARP_THREADS;
        dim3 grid_bwd(kv_seq_len/(1*kittens::TILE_ROW_DIM<bf16>*4), qo_heads, batch);
        bwd_attend_prep_ker<128, 1><<<grid_bwd, threads, mem_size, stream>>>(bwd_g);
    }

    using bwd_q_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_k_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_v_tile    =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_og_tile   =         st_bf<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_qg_tile   =         st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_kg_tile   =         st_fl<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_vg_tile   =         st_fl<bwd_attend_ker_tile_dims<128>::tile_h,    bwd_attend_ker_tile_dims<128>::tile_width>;
    using bwd_l_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_h>>;
    using bwd_d_tile    = row_vec<st_fl<bwd_attend_ker_tile_dims<128>::tile_h_qo, bwd_attend_ker_tile_dims<128>::tile_h>>;

    using bwd_q_global  = gl<bf16,  -1, -1, -1, -1, bwd_q_tile>;
    using bwd_k_global  = gl<bf16,  -1, -1, -1, -1, bwd_k_tile>;
    using bwd_v_global  = gl<bf16,  -1, -1, -1, -1, bwd_v_tile>;

    using bwd_og_global = gl<bf16,  -1, -1, -1, -1, bwd_og_tile>;

    using bwd_qg_global = gl<float, -1, -1, -1, -1, bwd_qg_tile>;
    using bwd_kg_global = gl<float, -1, -1, -1, -1, bwd_kg_tile>;
    using bwd_vg_global = gl<float, -1, -1, -1, -1, bwd_vg_tile>;

    using bwd_l_global  = gl<float, -1, -1, -1, -1, bwd_l_tile>;
    using bwd_d_global  = gl<float, -1, -1, -1, -1, bwd_d_tile>;

    using bwd_global_args = bwd_globals<128>;

    bwd_q_global  bwd_q_arg {d_q,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(q_seq_len), 128U};
    bwd_k_global  bwd_k_arg {d_k,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kv_seq_len), 128U};
    bwd_v_global  bwd_v_arg {d_v,  static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kv_seq_len), 128U};
    bwd_og_global bwd_og_arg{d_og, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(q_seq_len), 128U};
    bwd_qg_global bwd_qg_arg{d_qg, static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), static_cast<unsigned int>(q_seq_len), 128U};
    bwd_kg_global bwd_kg_arg{d_kg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kv_seq_len), 128U};
    bwd_vg_global bwd_vg_arg{d_vg, static_cast<unsigned int>(batch), static_cast<unsigned int>(kv_heads), static_cast<unsigned int>(kv_seq_len), 128U};
    bwd_l_global  bwd_l_arg {d_l,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(q_seq_len)};
    bwd_d_global  bwd_d_arg {d_d,  static_cast<unsigned int>(batch), static_cast<unsigned int>(qo_heads), 1U,   static_cast<unsigned int>(q_seq_len)};

    bwd_global_args bwd_global{
                    bwd_q_arg,
                    bwd_k_arg,
                    bwd_v_arg,
                    bwd_og_arg,
                    bwd_qg_arg,
                    bwd_kg_arg,
                    bwd_vg_arg,
                    bwd_l_arg,
                    bwd_d_arg,
                    static_cast<int>(q_seq_len),
                    static_cast<int>(hr)};

    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    const int ratio = kv_block_size / 64;
    const int small_block_ratio = q_block_size / 16;

    if (kv_block_size % 128 == 0) {
        constexpr int num_consumer = 2;
        constexpr int num_warpgoups = num_consumer + BWD_PRODUCER_WARPGROUPS;
        constexpr int num_workers = num_warpgoups * kittens::WARPGROUP_WARPS;

        dim3 grid_bwd_2(kv_seq_len / (num_consumer * 4 * kittens::TILE_ROW_DIM<bf16>), qo_heads, batch);
        auto threads = kittens::WARP_THREADS * num_workers;

        cudaFuncSetAttribute(
            bwd_attend_ker<128, num_consumer, num_warpgoups, num_workers>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            194000
        );
        cudaFuncSetAttribute(
            bwd_attend_ker<128, num_consumer, num_warpgoups, num_workers>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            85
        );

        bwd_attend_ker<128, num_consumer, num_warpgoups, num_workers><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global, ratio, block_mask_ptr, block_mask_dim,
                                                                                                               small_block_mask_ptr, use_small_block_mode, small_block_ratio);
    } else if (kv_block_size % 64 == 0) {
        constexpr int num_consumer = 1;
        constexpr int num_warpgoups = num_consumer + BWD_PRODUCER_WARPGROUPS;
        constexpr int num_workers = num_warpgoups * kittens::WARPGROUP_WARPS;

        dim3 grid_bwd_2(kv_seq_len / (num_consumer * 4 * kittens::TILE_ROW_DIM<bf16>), qo_heads, batch);
        auto threads = kittens::WARP_THREADS * num_workers;

        cudaFuncSetAttribute(
            bwd_attend_ker<128, num_consumer, num_warpgoups, num_workers>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            194000
        );
        cudaFuncSetAttribute(
            bwd_attend_ker<128, num_consumer, num_warpgoups, num_workers>,
            cudaFuncAttributePreferredSharedMemoryCarveout,
            85
        );

        bwd_attend_ker<128, num_consumer, num_warpgoups, num_workers><<<grid_bwd_2, threads, 194000, stream>>>(bwd_global, ratio, block_mask_ptr, block_mask_dim,
                                                                                                               small_block_mask_ptr, use_small_block_mode, small_block_ratio);
    }

    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaStreamSynchronize(stream);
    cudaDeviceSynchronize();

    return {qg, kg, vg};
}

#endif
