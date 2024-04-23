class Pipeline2x2Interleaved(object):
    def __init__(self):
        self.k_pipeline_src = self.gen_pipeline()

    def gen_pipeline(self):
        PIPELINE =  """
    buffer_load_dwordx4 v[v_gld_a1 + 0 : v_gld_a1 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0
    buffer_load_dwordx4 v[v_gld_a1 + 4 : v_gld_a1 + 7], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], s[s_offset_a + 0] offen offset:0
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    buffer_load_dwordx4 v[v_gld_b1 + 0 : v_gld_b1 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]

    ; store gld_a0 to lds
    s_waitcnt vmcnt(5)
    ds_write_b128 v[v_sst_offset_a0], v[v_gld_a0 : v_gld_a0 + 3], offset: 0
    buffer_load_dwordx4 v[v_gld_a0 + 0 : v_gld_a0 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0

    s_waitcnt vmcnt(5)
    ds_write_b128 v[v_sst_offset_a0], v[v_gld_a0 + 4 : v_gld_a0 + 7], offset: 64 * 8 * 2
    buffer_load_dwordx4 v[v_gld_a0 + 4 : v_gld_a0 + 7], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], s[s_offset_a + 0] offen offset:0

    ; dequant gld_b0
    s_waitcnt vmcnt(5)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 0

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 2, v_sel_b + 0, v_sub_magic_num, v_scale
    buffer_load_dwordx4 v[v_gld_b0 + 0 : v_gld_b0 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 1

    s_waitcnt lgkmcnt(0)
    s_barrier
   
    s_mov_b32 s[s_kitr], 32 * (1 + 0) ; 1 prefetch
    s_cmp_le_u32 s[s_k_per_cta], s[s_kitr]
    s_cbranch_scc1 label_gemm_rrr_loop_last_1

    s_mov_b32 s[s_kitr], 32 * (1 + 1)
    s_cmp_le_u32 s[s_k_per_cta], s[s_kitr]
    s_cbranch_scc1 label_gemm_rrr_loop_last_2

    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a0], offset: 0
    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b0], offset: 0 
label_gemm_rrr_loop_begin:
    ; load from lds and do mfma
    ;.mfma_wg1x1_w1x4_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c
    ; s_barrier
    ds_read_b128 v[v_sld_b1 + 0 : v_sld_b1 + 3], v[v_sld_offset_b0], offset: 64 * 8 * 2 * 1
    ds_read_b128 v[v_sld_a1 + 0 : v_sld_a1 + 3], v[v_sld_offset_a0], offset: 64 * 8 * 2 * 1
    ; global load n + 2
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]
    
    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 0 : v_c + 15]
    
    s_waitcnt vmcnt(5)
    ds_write_b128 v[v_sst_offset_a1], v[v_gld_a1 : v_gld_a1 + 3], offset: 0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 0 : v_c + 15]
    
    buffer_load_dwordx4 v[v_gld_a1 + 0 : v_gld_a1 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0
    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 16 : v_c + 31], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 16 : v_c + 31]
    
    s_waitcnt vmcnt(5)
    ds_write_b128 v[v_sst_offset_a1], v[v_gld_a1 + 4 : v_gld_a1 + 7], offset: 64 * 8 * 2
    buffer_load_dwordx4 v[v_gld_a1 + 4 : v_gld_a1 + 7], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], s[s_offset_a + 0] offen offset:0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 16 : v_c + 31], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 16 : v_c + 31]
 
    s_waitcnt vmcnt(5)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 32 : v_c + 47], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 32 : v_c + 47]

    ds_write_b128 v[v_sst_offset_b1], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 32 : v_c + 47], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 32 : v_c + 47]

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 2, v_sel_b + 0, v_sub_magic_num, v_scale
    buffer_load_dwordx4 v[v_gld_b1 + 0 : v_gld_b1 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 48 : v_c + 63], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 48 : v_c + 63]

    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a0 + 1], offset: (128 + 0) * 8 * 2 * 2 * 1
    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b0], offset: 128 * 8 * 2 * 2 * 1

    v_mfma_f32_32x32x8bf16_1k v[v_c + 48 : v_c + 63], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 48 : v_c + 63]

    ds_write_b128 v[v_sst_offset_b1], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 1

    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]

    s_waitcnt lgkmcnt(1)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 0 : v_c + 15]

    ds_read_b128 v[v_sld_b1 + 0 : v_sld_b1 + 3], v[v_sld_offset_b0], offset: 128 * 8 * 2 * 2 * 1 + 64 * 8 * 2 * 1
    ds_read_b128 v[v_sld_a1 + 0 : v_sld_a1 + 3], v[v_sld_offset_a0 + 1], offset: (128 + 0) * 8 * 2 * 2 * 1 + 64 * 8 * 2 * 1

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 0 : v_c + 15]

    s_waitcnt lgkmcnt(1)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 16 : v_c + 31], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 16 : v_c + 31]

    v_mfma_f32_32x32x8bf16_1k v[v_c + 16 : v_c + 31], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 16 : v_c + 31]

    s_waitcnt lgkmcnt(0)
    s_barrier

    v_mfma_f32_32x32x8bf16_1k v[v_c + 32 : v_c + 47], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 32 : v_c + 47]

    v_mfma_f32_32x32x8bf16_1k v[v_c + 32 : v_c + 47], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 32 : v_c + 47]

    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a1], offset: 0
    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b1], offset: 0
 
    v_mfma_f32_32x32x8bf16_1k v[v_c + 48 : v_c + 63], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 48 : v_c + 63]

    v_mfma_f32_32x32x8bf16_1k v[v_c + 48 : v_c + 63], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 48 : v_c + 63]

    ds_read_b128 v[v_sld_b1 + 0 : v_sld_b1 + 3], v[v_sld_offset_b1], offset: 64 * 8 * 2 * 1
    ds_read_b128 v[v_sld_a1 + 0 : v_sld_a1 + 3], v[v_sld_offset_a1], offset: 64 * 8 * 2 * 1

    s_waitcnt lgkmcnt(2)
   
    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 0 : v_c + 15]

    s_waitcnt vmcnt(5)
    ds_write_b128 v[v_sst_offset_a0], v[v_gld_a0 : v_gld_a0 + 3], offset: 0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 0 : v_c + 15]

    buffer_load_dwordx4 v[v_gld_a0 + 0 : v_gld_a0 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0
    s_waitcnt lgkmcnt(1)
 
    v_mfma_f32_32x32x8bf16_1k v[v_c + 16 : v_c + 31], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 16 : v_c + 31]

    v_mfma_f32_32x32x8bf16_1k v[v_c + 16 : v_c + 31], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 16 : v_c + 31]

    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a1 + 1], offset: (128 + 0) * 8 * 2 * 2 * 1

    v_mfma_f32_32x32x8bf16_1k v[v_c + 32 : v_c + 47], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 32 : v_c + 47]

    v_mfma_f32_32x32x8bf16_1k v[v_c + 32 : v_c + 47], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 32 : v_c + 47]

    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b1], offset: 128 * 8 * 2 * 2 * 1

    v_mfma_f32_32x32x8bf16_1k v[v_c + 48 : v_c + 63], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 48 : v_c + 63]

    v_mfma_f32_32x32x8bf16_1k v[v_c + 48 : v_c + 63], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 48 : v_c + 63]


    ds_read_b128 v[v_sld_b1 + 0 : v_sld_b1 + 3], v[v_sld_offset_b1], offset: 128 * 8 * 2 * 2 * 1 + 64 * 8 * 2 * 1
    ds_read_b128 v[v_sld_a1 + 0 : v_sld_a1 + 3], v[v_sld_offset_a1 + 1], offset: (128 + 0) * 8 * 2 * 2 * 1 + 64 * 8 * 2 * 1

    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 0 : v_c + 15]

    s_waitcnt vmcnt(5)
    ds_write_b128 v[v_sst_offset_a0], v[v_gld_a0 + 4 : v_gld_a0 + 7], offset: 64 * 8 * 2

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 0 : v_c + 15]

    buffer_load_dwordx4 v[v_gld_a0 + 4 : v_gld_a0 + 7], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], s[s_offset_a + 0] offen offset:0
    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 16 : v_c + 31], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 16 : v_c + 31]

    s_waitcnt vmcnt(5)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 0, v_sel_b + 0, v_sub_magic_num, v_scale

    v_mfma_f32_32x32x8bf16_1k v[v_c + 16 : v_c + 31], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 16 : v_c + 31]

    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 0
    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 32 : v_c + 47], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 32 : v_c + 47]

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 2, v_sel_b + 0, v_sub_magic_num, v_scale

    v_mfma_f32_32x32x8bf16_1k v[v_c + 32 : v_c + 47], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 32 : v_c + 47]

    buffer_load_dwordx4 v[v_gld_b0 + 0 : v_gld_b0 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 1

    v_mfma_f32_32x32x8bf16_1k v[v_c + 48 : v_c + 63], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 48 : v_c + 63]

    s_waitcnt lgkmcnt(0)
    s_barrier

    v_mfma_f32_32x32x8bf16_1k v[v_c + 48 : v_c + 63], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 48 : v_c + 63]


    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a0], offset: 0
    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b0], offset: 0 

    s_add_u32 s[s_kitr], 64, s[s_kitr] ; 64 * (1 + 1) 1 prefetch
    s_cmp_lt_u32 s[s_kitr], s[s_k_per_cta]
    s_cbranch_scc1 label_gemm_rrr_loop_begin

    s_sub_u32 s[s_kitr], s[s_kitr], 32
    s_cmp_lt_u32 s[s_kitr], s[s_k_per_cta]
    s_cbranch_scc1 label_gemm_rrr_loop_last_2
    
    s_branch label_gemm_rrr_loop_last_1

label_gemm_rrr_loop_last_2:
    ; load from lds and do mfma
    .mfma_wg2x2_w2x2_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c
    s_barrier
    
    ; store gld_a0 to lds
    s_waitcnt vmcnt(2)
    ds_write_b128 v[v_sst_offset_a0], v[v_gld_a1 : v_gld_a1 + 3], offset: 0

    s_waitcnt vmcnt(1)
    ds_write_b128 v[v_sst_offset_a0], v[v_gld_a1 + 4 : v_gld_a1 + 7], offset: 64 * 8 * 2

    ; dequant gld_b0
    s_waitcnt vmcnt(0)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 0

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 2, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 1

    s_waitcnt lgkmcnt(0)
    s_barrier

    ; load from lds and do mfma
    .mfma_wg2x2_w2x2_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c
    s_barrier

    s_branch label_write_out_c 
    
label_gemm_rrr_loop_last_1:
    ; load from lds and do mfma
    .mfma_wg2x2_w2x2_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c

"""

        return PIPELINE


