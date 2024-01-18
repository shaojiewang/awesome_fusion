


.macro .print v_val, s_out, s_bx, v_tid, v_offset
    ;s_mov_b64 exec, -1
    s_nop 64
    s_cmp_eq_u32 s[\s_bx], 0
    ;s_cbranch_scc0 L_endhere
    ;v_cmpx_eq_u32 0, v0
    v_lshlrev_b32 v[\v_offset], 3, v[\v_tid]
    s_waitcnt lgkmcnt(0)
    s_waitcnt vmcnt(0)
    global_store_dword v[\v_offset], v[\v_tid], s[\s_out:\s_out+1], offset:0x0
    global_store_dword v[\v_offset], v[\v_val], s[\s_out:\s_out+1], offset:0x0004
    ;s_mov_b64 exec, -1
;L_endhere:
    s_endpgm  
.endm

.macro .dequant_int8_1x8 v_tmp, v_base, v_gld_b, v_sel_b, v_sub_magic_num, v_scale
;.endm
;.macro fake0
    v_perm_b32 v[\v_tmp + 0], v[\v_base], v[\v_gld_b], v[\v_sel_b + 0]
    v_perm_b32 v[\v_tmp + 1], v[\v_base], v[\v_gld_b], v[\v_sel_b + 1]
    v_perm_b32 v[\v_tmp + 2], v[\v_base], v[\v_gld_b], v[\v_sel_b + 2]
    v_perm_b32 v[\v_tmp + 3], v[\v_base], v[\v_gld_b], v[\v_sel_b + 3]

    v_pk_add_f32 v[\v_tmp + 0 : \v_tmp + 1], v[\v_tmp + 0 : \v_tmp + 1], v[\v_sub_magic_num + 0 : \v_sub_magic_num + 1]
    v_pk_add_f32 v[\v_tmp + 2 : \v_tmp + 3], v[\v_tmp + 2 : \v_tmp + 3], v[\v_sub_magic_num + 0 : \v_sub_magic_num + 1]

    v_pk_mul_f32 v[\v_tmp + 0 : \v_tmp + 1], v[\v_scale + 0 : \v_scale + 1], v[\v_tmp + 0 : v_tmp + 1]
    v_pk_mul_f32 v[\v_tmp + 2 : \v_tmp + 3], v[\v_scale + 0 : \v_scale + 1], v[\v_tmp + 2 : v_tmp + 3]

    v_pack_b32_f16 v[\v_tmp + 0], v[\v_tmp + 0], v[\v_tmp + 1], op_sel: [1, 1]
    v_pack_b32_f16 v[\v_tmp + 1], v[\v_tmp + 2], v[\v_tmp + 3], op_sel: [1, 1]

    v_perm_b32 v[\v_tmp + 4], v[\v_base], v[\v_gld_b + 1], v[\v_sel_b + 0]
    v_perm_b32 v[\v_tmp + 5], v[\v_base], v[\v_gld_b + 1], v[\v_sel_b + 1]
    v_perm_b32 v[\v_tmp + 6], v[\v_base], v[\v_gld_b + 1], v[\v_sel_b + 2]
    v_perm_b32 v[\v_tmp + 7], v[\v_base], v[\v_gld_b + 1], v[\v_sel_b + 3]

    v_pk_add_f32 v[\v_tmp + 4 : \v_tmp + 5], v[\v_tmp + 4 : \v_tmp + 5], v[\v_sub_magic_num + 0 : \v_sub_magic_num + 1]
    v_pk_add_f32 v[\v_tmp + 6 : \v_tmp + 7], v[\v_tmp + 6 : \v_tmp + 7], v[\v_sub_magic_num + 0 : \v_sub_magic_num + 1]

    v_pk_mul_f32 v[\v_tmp + 4 : \v_tmp + 5], v[\v_scale + 0 : \v_scale + 1], v[\v_tmp + 4 : v_tmp + 5]
    v_pk_mul_f32 v[\v_tmp + 6 : \v_tmp + 7], v[\v_scale + 0 : \v_scale + 1], v[\v_tmp + 6 : v_tmp + 7]

    v_pack_b32_f16 v[\v_tmp + 2], v[\v_tmp + 4], v[\v_tmp + 5], op_sel: [1, 1]
    v_pack_b32_f16 v[\v_tmp + 3], v[\v_tmp + 6], v[\v_tmp + 7], op_sel: [1, 1]

.endm

.macro .mfma_wg1x1_w2x4_16x16x16bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a, v_sld_offset_b, v_c
;    .rept 8
;        v_fmac_f32 v0, v1, v2
;    .endr
;.endm
;.macro fake1
    ds_read_b128 v[\v_sld_a0 + 0 : \v_sld_a0 + 3], v[\v_sld_offset_a], offset: 0
    ds_read_b128 v[\v_sld_b0 + 0 : \v_sld_b0 + 3], v[\v_sld_offset_b], offset: 0 
    ds_read_b128 v[\v_sld_b1 + 0 : \v_sld_b1 + 3], v[\v_sld_offset_b], offset: 64 * 8 * 4 * 2 * 1
    ds_read_b128 v[\v_sld_a1 + 0 : \v_sld_a1 + 3], v[\v_sld_offset_a], offset: (32 + 1) * 8 * 4 * 2 * 1
    s_waitcnt lgkmcnt(0)

    v_mfma_f32_16x16x16bf16_1k v[\v_c + 0 : \v_c + 3], v[\v_sld_a0 + 0 : \v_sld_a0 + 1], v[\v_sld_b0 + 0 : \v_sld_b0 + 1], v[\v_c + 0 : \v_c + 3]
    s_setprio 1
    v_mfma_f32_16x16x16bf16_1k v[\v_c + 0 : \v_c + 3], v[\v_sld_a0 + 2 : \v_sld_a0 + 3], v[\v_sld_b0 + 2 : \v_sld_b0 + 3], v[\v_c + 0 : \v_c + 3]
    v_mfma_f32_16x16x16bf16_1k v[\v_c + 0 : \v_c + 3], v[\v_sld_a1 + 0 : \v_sld_a1 + 1], v[\v_sld_b1 + 0 : \v_sld_b1 + 1], v[\v_c + 0 : \v_c + 3]
    v_mfma_f32_16x16x16bf16_1k v[\v_c + 0 : \v_c + 3], v[\v_sld_a1 + 2 : \v_sld_a1 + 3], v[\v_sld_b1 + 2 : \v_sld_b1 + 3], v[\v_c + 0 : \v_c + 3]
    s_setprio 0
.endm

;kernel arguments OFFSET, shift in 1 byte
.set k_ptr_c,           0
.set k_ptr_a,           8
.set k_ptr_b,           16
.set k_ptr_scale,       24
.set k_m,               32
.set k_n,               36
.set k_k,               40
.set k_lda,             44
.set k_ldb,             48
.set k_ldc,             52
.set k_print,           56
.set k_end,             64

;sgpr
.set s_ka,              0
.set s_bx,              2
.set s_by,              3
.set s_ptr_c,           4
.set s_ptr_a,           8
.set s_ptr_b,           12
.set s_ptr_scale,       16
.set s_m,               20
.set s_n,               21
.set s_k,               22
.set s_lda,             23
.set s_ldb,             24
.set s_ldc,             25
.set s_print,           26
.set s_bs_a,            30
.set s_bs_b,            31
.set s_m_blocks,        32
.set s_m_idx,           33
.set s_n_idx,           34
.set s_offset_a,        35
.set s_offset_b,        36
.set s_kitr,            40
.set s_wave_id,         41
.set s_wave_im,         42
.set s_wave_in,         43
.set s_tmp,             64
.set s_end,             79

;vgpr
.set v_c,               0
.set v_sld_a0,          4
.set v_sld_b0,          8
.set v_sld_a1,          12
.set v_sld_b1,          16
.set v_gld_a0,          20
.set v_gld_a1,          22
.set v_gld_b0,          24
.set v_gld_b1,          26
.set v_lane_id,         60
.set v_offset_a_k0,     61
.set v_offset_a,        62
.set v_offset_b_k0,     63
.set v_offset_b,        64
.set v_lane_im,         65
.set v_lane_in,         66
.set v_sst_offset_c,    67
.set v_iak0,            68
.set v_im,              69
.set v_ibk0,            70
.set v_in,              71
.set v_sst_offset_a0,   72
.set v_sst_offset_a1,   73
.set v_sst_offset_b0,   74
.set v_sst_offset_b1,   75
.set v_sld_iak0,        76
.set v_sld_im,          77
.set v_sld_offset_a0,   78
.set v_sld_offset_a1,   79
.set v_sld_ibk0,        80
.set v_sld_in,          81
.set v_sld_offset_b0,   82
.set v_sld_offset_b1,   83
.set v_c_in,            84
.set v_c_im,            85
.set v_sld_offset_c,    86
.set v_gst_offset_c,    87
.set v_fp32_base,       88
.set v_sel_b,           89 ; total 4
.set v_sub_magic_num,   94
.set v_scale,           96
.set v_c_n_flag,        98
.set v_c_cur_m,         99
.set v_tid,             100
.set v_wave_id,         101
.set v_tmp,             104

.text
.global bf16gemm_rr8r_wg512_32x64x64_wg1x1_w2x4_16x16x16bf16_1k_pregld1
.p2align 8
.type bf16gemm_rr8r_wg512_32x64x64_wg1x1_w2x4_16x16x16bf16_1k_pregld1,@function
bf16gemm_rr8r_wg512_32x64x64_wg1x1_w2x4_16x16x16bf16_1k_pregld1:
    ; http://www.hsafoundation.com/html/Content/Runtime/Topics/02_Core/hsa_kernel_dispatch_packet_t.htm

    s_load_dwordx2 s[s_ptr_c:s_ptr_c+1], s[s_ka:s_ka+1], 0+k_ptr_c
    s_load_dwordx2 s[s_ptr_a:s_ptr_a+1], s[s_ka:s_ka+1], 0+k_ptr_a
    s_load_dwordx2 s[s_ptr_b:s_ptr_b+1], s[s_ka:s_ka+1], 0+k_ptr_b
    s_load_dwordx2 s[s_ptr_scale:s_ptr_scale+1], s[s_ka:s_ka+1], 0+k_ptr_scale

    s_load_dwordx4 s[s_m:s_m+3], s[s_ka:s_ka+1], 0+k_m
    s_load_dwordx4 s[s_ldb:s_ldb+3], s[s_ka:s_ka+1], 0+k_ldb
    
    v_mov_b32 v[v_tid], v0
    s_mov_b32 s[s_ptr_a + 3], 0x27000    
    s_mov_b32 s[s_ptr_b + 3], 0x27000    
    s_mov_b32 s[s_ptr_c + 3], 0x27000    
    s_mov_b32 s[s_ptr_scale + 3], 0x27000
    
    v_mov_b32 v[v_fp32_base], 0x4B000000
    v_mov_b32 v[v_sel_b + 0], 0x07060500
    v_mov_b32 v[v_sel_b + 1], 0x07060501
    v_mov_b32 v[v_sel_b + 2], 0x07060502
    v_mov_b32 v[v_sel_b + 3], 0x07060503

    v_cvt_f32_i32 v[v_sub_magic_num + 0], -8388736
    v_cvt_f32_i32 v[v_sub_magic_num + 1], -8388736

    s_waitcnt lgkmcnt(0)


    ; A and C matrix is bf16 datatype
    s_lshl_b32 s[s_lda], s[s_lda], 1
    s_lshl_b32 s[s_ldc], s[s_ldc], 1

    ; thread block mapping
    ; m block id: bid x
    ; n block id: bid y
    s_lshl_b32 s[s_m_idx], s[s_bx], 5
    s_lshl_b32 s[s_n_idx], s[s_by], 6
    
    ; load scale
    ; TODO: to avoid cache line waste
    ; Scale:
    ; thread vec: [n]         = [  1]
    ; block vec:  [k0, n, k1] = [  8, 64,  1]
    v_and_b32 v[v_tmp], v[v_tid], 63
    v_lshlrev_b32 v[v_tmp], 2, v[v_tmp]
    s_lshl_b32 s[s_tmp], s[s_n_idx], 2
    s_add_u32  s[s_ptr_scale], s[s_ptr_scale], s[s_tmp]
    s_addc_u32 s[s_ptr_scale + 1], s[s_ptr_scale + 1], 0
    s_lshl_b32 s[s_ptr_scale + 2], s[s_n], 2
    s_sub_i32 s[s_ptr_scale + 2], s[s_ptr_scale + 2], s[s_tmp]
    buffer_load_dword v[v_scale], v[v_tmp], s[s_ptr_scale : s_ptr_scale + 3], 0 offen offset: 0

    ; load A/B matrix
    ; A:
    ; thread vec: [ak0, m, ak1] = [  1,  1,  4]
    ; block vec:  [ak0, m, ak1] = [ 16, 32,  1]
    ; B:
    ; thread vec: [bk0, n, bk1] = [  1,  1,  8]
    ; block vec:  [bk0, n, bk1] = [  8, 64,  1]

    ; A thread block offset
    v_and_b32 v[v_iak0], v[v_tid], 15
    v_lshrrev_b32 v[v_im], 4, v[v_tid]
    v_lshlrev_b32 v[v_tmp], 3, v[v_iak0]
    v_mad_u32_u24 v[v_offset_a], v[v_im], s[s_lda], v[v_tmp]
    ; A grid offset
    s_mul_i32 s[s_tmp], s[s_m_idx], s[s_lda]
    s_add_u32  s[s_ptr_a], s[s_ptr_a], s[s_tmp]
    s_addc_u32 s[s_ptr_a + 1], s[s_ptr_a + 1], 0
    ; prefetch load A
    s_mul_i32 s[s_ptr_a + 2], s[s_m], s[s_lda]
    s_sub_i32 s[s_ptr_a + 2], s[s_ptr_a + 2], s[s_tmp]
    
    buffer_load_dwordx2 v[v_gld_a0 + 0 : v_gld_a0 + 1], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0
    s_mov_b32 s[s_bs_a], 128
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]

    ; B thread block offset
    v_and_b32 v[v_in], v[v_tid], 63
    v_lshrrev_b32 v[v_ibk0], 6, v[v_tid]
    v_lshlrev_b32 v[v_tmp], 3, v[v_in]
    ; k0 offset = ldb
    v_mad_u32_u24 v[v_offset_b], v[v_ibk0], s[s_ldb], v[v_tmp]
    ; B grid offset
    s_lshl_b32 s[s_tmp], s[s_n_idx], 3
    s_add_u32  s[s_ptr_b], s[s_ptr_b], s[s_tmp]
    s_addc_u32 s[s_ptr_b + 1], s[s_ptr_b + 1], 0
    ; prefetch load B
    s_lshr_b32 s[s_tmp + 1], s[s_ldb], 3
    s_mul_i32 s[s_ptr_b + 2], s[s_k], s[s_tmp + 1]
    s_sub_i32 s[s_ptr_b + 2], s[s_ptr_b + 2], s[s_tmp]

    buffer_load_dwordx2 v[v_gld_b0 + 0 : v_gld_b0 + 1], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    s_lshl_b32 s[s_bs_b], s[s_ldb], 3
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]

    ; store C offset
    ; vgpr to lds
    ; vgpr_group  = 4
    ; wave_id = tid / wave_size
    ; lane_id = tid % wave_size
    ; lane_in = lane_id % inst_n = tid % inst_n
    ; lane_im = lane_id / inst_n * vgpr_group
    ; wave_n = block_n / inst_n
    ; wave_m = block_m / inst_m
    ; wave_in = wave_id % wave_n
    ; wave_im = wave_id / wave_n

    ; wave id
    v_lshrrev_b32 v[v_wave_id], 6, v[v_tid]
    v_readfirstlane_b32 s[s_wave_id], v[v_wave_id]
    s_lshr_b32 s[s_wave_im], s[s_wave_id], 2
    s_and_b32  s[s_wave_in], s[s_wave_id], 3
    s_lshl_b32 s[s_wave_im], s[s_wave_im], 4
    s_lshl_b32 s[s_wave_in], s[s_wave_in], 4

    ; lane id
    v_and_b32 v[v_lane_id], 63, v[v_tid]
    v_and_b32 v[v_lane_in], 15, v[v_tid] 
    v_lshrrev_b32 v[v_lane_im], 4, v[v_lane_id]
    v_lshlrev_b32 v[v_lane_im], 2, v[v_lane_im]

    ; sst offset C
    ; m_offset = (wave_im + lane_im) * block_n
    ; n_offset = wave_n + lane_in
    ; sst_c_offset = m_offset + n_offset
    v_add_lshl_u32 v[v_sst_offset_c], v[v_lane_im], s[s_wave_im], 6
    v_add_u32 v[v_tmp], v[v_lane_in], s[s_wave_in]
    v_add_lshl_u32 v[v_sst_offset_c], v[v_tmp], v[v_sst_offset_c], 1

    ; sld/gst offset C
    ; c_in = tid % (block_n / vec_c_n)
    ; c_im = tid / (block_n / vec_c_n)
    ; sld_c_offset = c_in * vec_c_n + c_im * block_n
    ; gst_c_offset = c_in * vec_c_n + c_im * ldc
    v_and_b32 v[v_c_in], 15, v[v_tid]
    v_lshrrev_b32 v[v_c_im], 4, v[v_tid]
    v_lshlrev_b32 v[v_tmp], 3, v[v_c_in]
    v_lshl_add_u32 v[v_sld_offset_c], v[v_c_im], 7, v[v_tmp]
    v_mul_lo_u32 v[v_tmp + 1], v[v_c_im], s[s_ldc]
    v_add_u32 v[v_gst_offset_c], v[v_tmp + 1], v[v_tmp]
    ; c grid pointer
    s_mul_i32 s[s_tmp], s[s_m_idx], s[s_ldc]
    s_lshl_b32 s[s_tmp + 2], s[s_n_idx], 1
    s_add_u32 s[s_tmp + 1], s[s_tmp + 2], s[s_tmp]
    s_add_u32 s[s_ptr_c], s[s_ptr_c], s[s_tmp + 1]
    s_addc_u32 s[s_ptr_c + 1], s[s_ptr_c + 1], 0
    s_mul_i32 s[s_ptr_c + 2], s[s_m], s[s_ldc]
    ; c n flag
    v_lshl_add_u32 v[v_tmp], v[v_c_in], 3, s[s_n_idx]
    v_cmp_gt_u32 vcc, s[s_n], v[v_c_in]
    v_cndmask_b32 v[v_c_n_flag],  0, 1, vcc
    

    ; store A to shared mem offset
    ; sst_iak0 = iak0 * (block_m + pad) * ak1
    ; sst_offset_a = sst_iak0 + v_im * 8
    v_lshlrev_b32 v[v_tmp], 4, v[v_im]
    v_mov_b32 v[v_tmp + 1], (32 + 1) * 8 * 2
    v_lshrrev_b32 v[v_tmp + 2], 1, v[v_iak0]
    v_and_b32 v[v_tmp + 3], 1, v[v_iak0]
    v_lshlrev_b32 v[v_tmp + 3], 3, v[v_tmp + 3]
    v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp + 3]
    v_mad_u32_u24 v[v_sst_offset_a0], v[v_tmp + 2], v[v_tmp + 1], v[v_tmp]
    v_mov_b32 v[v_tmp], 0x4000
    v_xor_b32 v[v_sst_offset_a1], v[v_tmp], v[v_sst_offset_a0]

    ; store B to shared mem offset. when B is stored to shared mem, B datatype is bf16/fp16
    ; sst_in = v_in * bk1 * n1 = v_in * 8 * 1
    ; sst_ibk0 = v_ibk0 * block_n * bk1 = v_ibk0 * 64 * 8
    ; sst_offset_b = sst_in + sst_ibk0
    ; padding = sst_offset_b / 64 * 8
    ; sst_offset_b = sst_offset_b + padding
    v_lshlrev_b32 v[v_tmp], 3, v[v_in]
    v_lshlrev_b32 v[v_tmp + 1], 9, v[v_ibk0]
    v_add_u32 v[v_sst_offset_b0], v[v_tmp], v[v_tmp + 1]
    ; v_lshrrev_b32 v[v_tmp], 6, v[v_sst_offset_b]
    ; v_lshl_add_u32 v[v_sst_offset_b], v[v_tmp], 3, v[v_sst_offset_b] 
    v_lshlrev_b32 v[v_sst_offset_b0], 1, v[v_sst_offset_b0]
    v_mov_b32 v[v_tmp], 4224
    v_add_u32 v[v_sst_offset_b0], v[v_sst_offset_b0], v[v_tmp]
    v_mov_b32 v[v_tmp], 0x4000
    v_xor_b32 v[v_sst_offset_b1], v[v_tmp], v[v_sst_offset_b0]

    ; load A to shared mem offset
    ; sld_iak0 = laneid / inst_m * ((block_m + pad) * ak1)
    ; sld_im = lane_id % inst_m + wave_im
    ; sld_offset_a = sld_im * ak1 + sld_iak0
    v_lshrrev_b32 v[v_sld_iak0], 4, v[v_lane_id]
    v_mov_b32 v[v_tmp], (32 + 1) * 8
    v_mul_lo_u32 v[v_sld_iak0], v[v_tmp], v[v_sld_iak0] 
    v_and_b32 v[v_sld_im], 15, v[v_lane_id]
    v_add_lshl_u32 v[v_sld_im], v[v_sld_im], s[s_wave_im], 3
    v_add_lshl_u32 v[v_sld_offset_a0], v[v_sld_iak0], v[v_sld_im], 1
    v_mov_b32 v[v_tmp], 0x4000
    v_xor_b32 v[v_sld_offset_a1], v[v_tmp], v[v_sld_offset_a0]

    ; load B to shared mem offset
    ; k1 = max(ak1, bk1)
    ; sld_ibk0 = laneid / inst_n * (block_n * k1)
    ; sld_in = laneid % inst_n + wave_in
    ; sld_offset_b = sld_ibk0 + sld_in * bk1
    ; padding = sld_offset_b / 64 * 8
    ; sld_offset_b = padding + sld_offset_b
    v_lshrrev_b32 v[v_sld_ibk0], 4, v[v_lane_id]
    v_lshlrev_b32 v[v_sld_ibk0], 9, v[v_sld_ibk0]
    v_and_b32 v[v_sld_in], 15, v[v_lane_id]
    v_add_lshl_u32 v[v_sld_in], v[v_sld_in], s[s_wave_in], 3
    v_add_u32 v[v_sld_offset_b0], v[v_sld_in], v[v_sld_ibk0]
    v_lshlrev_b32 v[v_sld_offset_b0], 1, v[v_sld_offset_b0]
    v_mov_b32 v[v_tmp], 4224;(32 + 1) * 8 * 8 * 2
    v_add_u32 v[v_sld_offset_b0], v[v_sld_offset_b0], v[v_tmp]
    v_mov_b32 v[v_tmp], 0x4000
    v_xor_b32 v[v_sld_offset_b1], v[v_tmp], v[v_sld_offset_b0]

    ; double scale 
    s_waitcnt vmcnt(2)
    v_mov_b32 v[v_scale + 1], v[v_scale + 0]

    ; clear C vgpr
    .cnt = 0
    .rept 4
        v_mov_b32 v[v_c + .cnt], 0
        .cnt = .cnt + 1
    .endr

    s_mov_b32 s[s_kitr], 64 * (1 + 0) ; 1 prefetch
    s_cmp_le_u32 s[s_k], s[s_kitr]
    s_cbranch_scc1 label_gemm_rrr_loop_last_1

    buffer_load_dwordx2 v[v_gld_a1 + 0 : v_gld_a1 + 1], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0
    buffer_load_dwordx2 v[v_gld_b1 + 0 : v_gld_b1 + 1], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]

    s_mov_b32 s[s_kitr], 64 * (1 + 1)
    s_cmp_le_u32 s[s_k], s[s_kitr]
    s_cbranch_scc1 label_gemm_rrr_loop_last_2

label_gemm_rrr_loop_begin:
    ; global load n + 1
    
    ; store gld_a0 to lds
    s_waitcnt vmcnt(3)
    ds_write_b64 v[v_sst_offset_a0], v[v_gld_a0 : v_gld_a0 + 1], offset: 0
    buffer_load_dwordx2 v[v_gld_a0 + 0 : v_gld_a0 + 1], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0

    ; dequant gld_b0
    s_waitcnt vmcnt(3)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    buffer_load_dwordx2 v[v_gld_b0 + 0 : v_gld_b0 + 1], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 64 * 16 * 2 * 0

    s_waitcnt lgkmcnt(0)
    s_barrier
   
    ; load from lds and do mfma
    .mfma_wg1x1_w2x4_16x16x16bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c
    s_barrier
    
    ; global load n + 2
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]
    
    ; store gld_a0 to lds
    s_waitcnt vmcnt(3)
    ds_write_b64 v[v_sst_offset_a1], v[v_gld_a1 : v_gld_a1 + 1], offset: 0
    buffer_load_dwordx2 v[v_gld_a1 + 0 : v_gld_a1 + 1], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0

    ; dequant gld_b0
    s_waitcnt vmcnt(3)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    buffer_load_dwordx2 v[v_gld_b1 + 0 : v_gld_b1 + 1], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    ds_write_b128 v[v_sst_offset_b1], v[v_tmp : v_tmp + 3], offset: 64 * 16 * 2 * 0
    
    s_waitcnt lgkmcnt(0)
    s_barrier

    ; load from lds and do mfma
    .mfma_wg1x1_w2x4_16x16x16bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a1, v_sld_offset_b1, v_c
    s_barrier
 
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]

    s_add_u32 s[s_kitr], 128, s[s_kitr] ; 64 * (1 + 1) 1 prefetch
    s_cmp_lt_u32 s[s_kitr], s[s_k]
    s_cbranch_scc1 label_gemm_rrr_loop_begin

    s_sub_u32 s[s_kitr], s[s_kitr], 64
    s_cmp_lt_u32 s[s_kitr], s[s_k]
    s_cbranch_scc1 label_gemm_rrr_loop_last_2
    
    s_branch label_gemm_rrr_loop_last_1

label_gemm_rrr_loop_last_2:
    ; global load n + 1
    ;buffer_load_dwordx2 v[v_gld_a1 + 0 : v_gld_a1 + 1], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0
    ;v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    ;buffer_load_dwordx2 v[v_gld_b1 + 0 : v_gld_b1 + 1], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    ;v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]
    
    ; store gld_a0 to lds
    s_waitcnt vmcnt(3)
    ds_write_b64 v[v_sst_offset_a0], v[v_gld_a0 : v_gld_a0 + 1], offset: 0

    ; dequant gld_b0
    s_waitcnt vmcnt(2)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 64 * 16 * 2 * 0
    
    s_waitcnt lgkmcnt(0)
    s_barrier

    ; load from lds and do mfma
    .mfma_wg1x1_w2x4_16x16x16bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c
    s_barrier
    
    ; store gld_a1 to lds
    s_waitcnt vmcnt(1)
    ds_write_b64 v[v_sst_offset_a1], v[v_gld_a1 : v_gld_a1 + 1], offset: 0

    ; dequant gld_b1
    s_waitcnt vmcnt(0)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b1], v[v_tmp : v_tmp + 3], offset: 64 * 16 * 2 * 0
    
    s_waitcnt lgkmcnt(0)
    s_barrier

    ; load from lds and do mfma
    .mfma_wg1x1_w2x4_16x16x16bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a1, v_sld_offset_b1, v_c
    s_barrier

    s_branch label_write_out_c 
    
label_gemm_rrr_loop_last_1:
    ; store gld_a0 to lds
    s_waitcnt vmcnt(1)
    ds_write_b64 v[v_sst_offset_a0], v[v_gld_a0 : v_gld_a0 + 1], offset: 0

    ; dequant gld_b0
    s_waitcnt vmcnt(0)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 64 * 16 * 2 * 0
    
    s_waitcnt lgkmcnt(0)
    s_barrier

    ; load from lds and do mfma
    .mfma_wg1x1_w2x4_16x16x16bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c

label_write_out_c:
    s_nop 15
    s_barrier
    ; rtz mode (not so accurate)

    ; store to lds
    ; within 1 inst group
    ; imm_offset = block_n * sizeof(datatype) * v_groups * n_per_vgpr * i_inst (64 * 2 * 4 * 2)
    ; within 1 vgpr group
    ; imm_offset = block_n * sizeof(datatype) * i_vgpr (64 * 2)
    ; imm_offset = block_n * sizeof(datatype) * v_groups * n_per_vgpr * i_inst + block_n * sizeof(datatype) * i_vgpr
    .v_c_inst_cnt = 0
    .rept 1
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 0], offset: 64 * 2 * 4 * 2 * .v_c_inst_cnt + 64 * 2 * 0
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 1], offset: 64 * 2 * 4 * 2 * .v_c_inst_cnt + 64 * 2 * 1
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 2], offset: 64 * 2 * 4 * 2 * .v_c_inst_cnt + 64 * 2 * 2
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 3], offset: 64 * 2 * 4 * 2 * .v_c_inst_cnt + 64 * 2 * 3
        .v_c_inst_cnt = .v_c_inst_cnt + 1
    .endr
    s_waitcnt lgkmcnt(0)
    s_barrier

    ; load from lds
    ; imm_offset = 16 * threadim.x * i
    ds_read_b64 v[v_c + 0 : v_c + 1], v[v_sld_offset_c], offset: 16 * 128 * 0

    s_mov_b32 s[s_tmp], 0
    s_waitcnt lgkmcnt(0)
    s_barrier

    ; store res to global
    v_cmpx_eq_u32 vcc, 1, v[v_c_n_flag]    
    buffer_store_dwordx2 v[v_c + 0 : v_c + 1], v[v_gst_offset_c], s[s_ptr_c + 0 : s_ptr_c + 3], s[s_tmp] offen offset: 0
    s_mov_b64 exec, -1
    
    
    ; .print v_offset_a, s_print, s_bx, v_tid, v_tmp + 7
    

    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel bf16gemm_rr8r_wg512_32x64x64_wg1x1_w2x4_16x16x16bf16_1k_pregld1
    .amdhsa_group_segment_fixed_size 32768
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 128
    .amdhsa_next_free_sgpr 80
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_accum_offset 64
    # .amdhsa_wavefront_size32 1
    # .amdhsa_workgroup_processor_mode 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: bf16gemm_rr8r_wg512_32x64x64_wg1x1_w2x4_16x16x16bf16_1k_pregld1
    .symbol: bf16gemm_rr8r_wg512_32x64x64_wg1x1_w2x4_16x16x16bf16_1k_pregld1.kd
    .sgpr_count: 79
    .vgpr_count: 128
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [512, 1, 1]
    .max_flat_workgroup_size: 512
    .args:
    - { .name: ptr_c,           .size: 8, .offset:   0, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: false}
    - { .name: ptr_a,           .size: 8, .offset:   8, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: true }
    - { .name: ptr_b,           .size: 8, .offset:  16, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: true }
    - { .name: ptr_scale,       .size: 8, .offset:  24, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true}
    - { .name: m,               .size: 4, .offset:  32, .value_kind: by_value, .value_type: i32}
    - { .name: n,               .size: 4, .offset:  36, .value_kind: by_value, .value_type: i32}
    - { .name: k,               .size: 4, .offset:  40, .value_kind: by_value, .value_type: i32}
    - { .name: lda,             .size: 4, .offset:  44, .value_kind: by_value, .value_type: i32}
    - { .name: ldb,             .size: 4, .offset:  48, .value_kind: by_value, .value_type: i32}
    - { .name: ldc,             .size: 4, .offset:  52, .value_kind: by_value, .value_type: i32}
    - { .name: print,           .size: 8, .offset:  56, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false}
...
.end_amdgpu_metadata

