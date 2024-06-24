
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
    s_waitcnt vmcnt(0)
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

    ;v_pk_fma_f32 v[\v_tmp + 0 : \v_tmp + 1], v[\v_tmp + 0 : \v_tmp + 1], v[\v_scale + 0 : \v_scale + 1], v[\v_sub_magic_num + 0 : \v_sub_magic_num + 1]
    ;v_pk_fma_f32 v[\v_tmp + 2 : \v_tmp + 3], v[\v_tmp + 2 : \v_tmp + 3], v[\v_scale + 0 : \v_scale + 1], v[\v_sub_magic_num + 0 : \v_sub_magic_num + 1]

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

    ;v_pk_fma_f32 v[\v_tmp + 4 : \v_tmp + 5], v[\v_tmp + 4 : \v_tmp + 5], v[\v_scale + 0 : \v_scale + 1], v[\v_sub_magic_num + 0 : \v_sub_magic_num + 1]
    ;v_pk_fma_f32 v[\v_tmp + 6 : \v_tmp + 7], v[\v_tmp + 6 : \v_tmp + 7], v[\v_scale + 0 : \v_scale + 1], v[\v_sub_magic_num + 0 : \v_sub_magic_num + 1]
    
    v_pack_b32_f16 v[\v_tmp + 2], v[\v_tmp + 4], v[\v_tmp + 5], op_sel: [1, 1]
    v_pack_b32_f16 v[\v_tmp + 3], v[\v_tmp + 6], v[\v_tmp + 7], op_sel: [1, 1]

.endm

.macro .mfma_wg1x1_w1x4_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a, v_sld_offset_b, v_c
;    .rept 8
;        v_fmac_f32 v0, v1, v2
;    .endr
;.endm
;.macro fake1
    ds_read_b128 v[\v_sld_a0 + 0 : \v_sld_a0 + 3], v[\v_sld_offset_a], offset: 0
    ds_read_b128 v[\v_sld_b0 + 0 : \v_sld_b0 + 3], v[\v_sld_offset_b], offset: 0 
    ds_read_b128 v[\v_sld_b1 + 0 : \v_sld_b1 + 3], v[\v_sld_offset_b], offset: 128 * 8 * 2 * 2 * 1
    ds_read_b128 v[\v_sld_a1 + 0 : \v_sld_a1 + 3], v[\v_sld_offset_a + 1], offset: (32 + 0) * 8 * 2 * 2 * 1
    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[\v_c + 0 : \v_c + 15], v[\v_sld_a0 + 0 : \v_sld_a0 + 1], v[\v_sld_b0 + 0 : \v_sld_b0 + 1], v[\v_c + 0 : \v_c + 15]
    ; s_setprio 1
    v_mfma_f32_32x32x8bf16_1k v[\v_c + 0 : \v_c + 15], v[\v_sld_a0 + 2 : \v_sld_a0 + 3], v[\v_sld_b0 + 2 : \v_sld_b0 + 3], v[\v_c + 0 : \v_c + 15]
    ds_read_b128 v[\v_sld_a0 + 0 : \v_sld_a0 + 3], v[\v_sld_offset_a + 2], offset: (32 + 0) * 8 * 2 * 2 * 2
    ds_read_b128 v[\v_sld_b0 + 0 : \v_sld_b0 + 3], v[\v_sld_offset_b], offset: 128 * 8 * 2 * 2 * 2
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x8bf16_1k v[\v_c + 0 : \v_c + 15], v[\v_sld_a1 + 0 : \v_sld_a1 + 1], v[\v_sld_b1 + 0 : \v_sld_b1 + 1], v[\v_c + 0 : \v_c + 15]
    v_mfma_f32_32x32x8bf16_1k v[\v_c + 0 : \v_c + 15], v[\v_sld_a1 + 2 : \v_sld_a1 + 3], v[\v_sld_b1 + 2 : \v_sld_b1 + 3], v[\v_c + 0 : \v_c + 15]
    ; s_setprio 0
    
    ds_read_b128 v[\v_sld_b1 + 0 : \v_sld_b1 + 3], v[\v_sld_offset_b], offset: 128 * 8 * 2 * 2 * 3
    ds_read_b128 v[\v_sld_a1 + 0 : \v_sld_a1 + 3], v[\v_sld_offset_a + 3], offset: (32 + 0) * 8 * 2 * 2 * 3
    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[\v_c + 0 : \v_c + 15], v[\v_sld_a0 + 0 : \v_sld_a0 + 1], v[\v_sld_b0 + 0 : \v_sld_b0 + 1], v[\v_c + 0 : \v_c + 15]
    ;s_setprio 1
    v_mfma_f32_32x32x8bf16_1k v[\v_c + 0 : \v_c + 15], v[\v_sld_a0 + 2 : \v_sld_a0 + 3], v[\v_sld_b0 + 2 : \v_sld_b0 + 3], v[\v_c + 0 : \v_c + 15]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x8bf16_1k v[\v_c + 0 : \v_c + 15], v[\v_sld_a1 + 0 : \v_sld_a1 + 1], v[\v_sld_b1 + 0 : \v_sld_b1 + 1], v[\v_c + 0 : \v_c + 15]
    v_mfma_f32_32x32x8bf16_1k v[\v_c + 0 : \v_c + 15], v[\v_sld_a1 + 2 : \v_sld_a1 + 3], v[\v_sld_b1 + 2 : \v_sld_b1 + 3], v[\v_c + 0 : \v_c + 15]
    ;s_setprio 0
.endm
;kernel arguments OFFSET, shift in 1 byte
.set k_ptr_c, 0
.set k_ptr_a, 8
.set k_ptr_b, 16
.set k_ptr_scale, 24
.set k_m, 32
.set k_n, 36
.set k_k, 40
.set k_lda, 44
.set k_ldb, 48
.set k_ldc, 52
.set k_k_per_cta, 56
.set k_print, 60
.set k_multigpu_barrier_flag, 68
.set k_local_flag, 72
.set k_world_barrier, 80
.set k_local_out, 144
.set k_peer_comm_buffer, 152
.set k_local_rank, 160

;sgpr
.set s_ka, 0
.set s_bx, 2
.set s_by, 3
.set s_bz, 4
.set s_ptr_c, 8
.set s_ptr_a, 12
.set s_ptr_b, 16
.set s_ptr_scale, 20
.set s_m, 24
.set s_n, 25
.set s_k, 26
.set s_lda, 27
.set s_ldb, 28
.set s_ldc, 29
.set s_k_per_cta, 30
.set s_print, 32
.set s_local_flag, 34
.set s_world_barrier, 36
.set s_local_barrier, 38
.set s_local_out, 40
.set s_peer_comm_buffer, 42
.set s_bs_a, 44
.set s_bs_b, 45
.set s_m_blocks, 46
.set s_m_idx, 47
.set s_n_idx, 48
.set s_offset_a, 50
.set s_offset_b, 54
.set s_kitr, 55
.set s_wave_id, 56
.set s_wave_im, 57
.set s_wave_in, 58
.set s_k_idx, 59
.set s_offset_local_flag, 60
.set s_flag, 61
.set s_flag_checker, 62
.set s_multigpu_barrier_flag, 63
.set s_local_rank, 64
.set s_barrier_flag, 66
.set s_tmp, 80

;vgpr
.set v_c, 0
.set v_sld_a0, 16
.set v_sld_b0, 20
.set v_sld_a1, 24
.set v_sld_b1, 28
.set v_gld_a0, 32
.set v_gld_a1, 36
.set v_gld_b0, 40
.set v_gld_b1, 48
.set v_lane_id, 56
.set v_offset_a_k0, 57
.set v_offset_a, 58
.set v_offset_b_k0, 59
.set v_offset_b, 60
.set v_lane_im, 61
.set v_lane_in, 62
.set v_sst_offset_c, 63
.set v_iak0, 64
.set v_im, 65
.set v_ibk0, 66
.set v_in, 67
.set v_sst_offset_a0, 68
.set v_sst_offset_a1, 69
.set v_sst_offset_b0, 70
.set v_sst_offset_b1, 71
.set v_sld_iak0, 72
.set v_sld_im, 73
.set v_sld_offset_a0, 76
.set v_sld_offset_a1, 80
.set v_sld_ibk0, 84
.set v_sld_in, 85
.set v_sld_offset_b0, 86
.set v_sld_offset_b1, 87
.set v_c_in, 88
.set v_c_im, 89
.set v_sld_offset_c, 90
.set v_gst_offset_c, 91
.set v_fp32_base, 92
.set v_sel_b, 96
.set v_sub_magic_num, 100
.set v_scale, 102
.set v_c_n_flag, 104
.set v_c_cur_m, 105
.set v_tid, 106
.set v_wave_id, 107
.set v_tmp, 108
; local flag (can reuse the a/b/c vgprs)
.set v_imm, 0
.set v_flag, 1
.set v_flag_load, 2
.set v_offset_flag, 3
.set v_barrier_offset, 4
.set v_local_barrier_offset, 5
.set v_barrier_addr, 6
.set v_local_rank, 8
.set v_barrier_flag_check, 9
.set v_barrier_flag, 10


.text
.global bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk
.p2align 8
.type bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk,@function
bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk: 
    ; http://www.hsafoundation.com/html/Content/Runtime/Topics/02_Core/hsa_kernel_dispatch_packet_t.htm

    s_load_dwordx2 s[s_ptr_c:s_ptr_c+1], s[s_ka:s_ka+1], 0+k_ptr_c
    s_load_dwordx2 s[s_ptr_a:s_ptr_a+1], s[s_ka:s_ka+1], 0+k_ptr_a
    s_load_dwordx2 s[s_ptr_b:s_ptr_b+1], s[s_ka:s_ka+1], 0+k_ptr_b
    s_load_dwordx2 s[s_ptr_scale:s_ptr_scale+1], s[s_ka:s_ka+1], 0+k_ptr_scale
    s_load_dwordx2 s[s_print:s_print+1], s[s_ka:s_ka+1], 0+k_print
    s_load_dword s[s_multigpu_barrier_flag], s[s_ka:s_ka+1], 0+k_multigpu_barrier_flag
    s_load_dwordx2 s[s_local_flag:s_local_flag+1], s[s_ka:s_ka+1], 0+k_local_flag
    ; s_load_dwordx2 s[s_world_barrier:s_world_barrier+1], s[s_ka:s_ka+1], 0+k_world_barrier
    s_load_dwordx2 s[s_local_out:s_local_out+1], s[s_ka:s_ka+1], 0+k_local_out
    s_load_dwordx2 s[s_peer_comm_buffer:s_peer_comm_buffer+1], s[s_ka:s_ka+1], 0+k_peer_comm_buffer
    s_load_dwordx2 s[s_local_rank:s_local_rank+1], s[s_ka:s_ka+1], 0+k_local_rank

    s_load_dwordx4 s[s_m:s_m+3], s[s_ka:s_ka+1], 0+k_m
    s_load_dwordx2 s[s_ldb:s_ldb+1], s[s_ka:s_ka+1], 0+k_ldb
    s_load_dword s[s_k_per_cta], s[s_ka:s_ka+1], 0+k_k_per_cta
    
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


    ; load local barrier to do sync
    s_lshl_b64 s[s_tmp : s_tmp + 1], s[s_local_rank:s_local_rank+1], 3
    s_add_u32 s[s_tmp], s[s_ka], s[s_tmp]
    s_addc_u32 s[s_tmp + 1], s[s_ka + 1], s[s_tmp + 1]
    s_load_dwordx2 s[s_local_barrier : s_local_barrier + 1], s[s_tmp : s_tmp + 1], 0+k_world_barrier

    s_waitcnt lgkmcnt(0)

    ; adjust lda/b/c according to the datatypes
    s_lshl_b32 s[s_lda], s[s_lda], 1
    s_lshl_b32 s[s_ldc], s[s_ldc], 1

    ; thread block mapping
    ; m block id: bid x
    ; n block id: bid y
    ; k block id: bid z
    s_mul_i32 s[s_m_idx], s[s_by], 32
    s_mul_i32 s[s_n_idx], s[s_bx], 128
    s_mul_i32 s[s_k_idx], s[s_bz], s[s_k_per_cta]


    ; load scale
    ; TODO: to avoid cache line waste
    ; Scale:
    ; thread vec: [n]         = [1]
    ; block vec:  [k0, n, k1] = [2,128,1]
    v_mov_b32 v[v_tmp], 127
    v_and_b32 v[v_tmp], v[v_tid], v[v_tmp]
    v_lshlrev_b32 v[v_tmp], 2, v[v_tmp]
    s_lshl_b32 s[s_tmp], s[s_n_idx], 2
    s_add_u32  s[s_ptr_scale], s[s_ptr_scale], s[s_tmp]
    s_addc_u32 s[s_ptr_scale + 1], s[s_ptr_scale + 1], 0
    s_lshl_b32 s[s_ptr_scale + 2], s[s_n], 2
    s_sub_i32 s[s_ptr_scale + 2], s[s_ptr_scale + 2], s[s_tmp]

    buffer_load_dword v[v_scale], v[v_tmp], s[s_ptr_scale : s_ptr_scale + 3], 0 offen offset: 0

    ; load A matrix
    ; A:
    ; thread vec: [ak0, m, ak1] = [1, 1, 8]
    ; block vec:  [ak0, m, ak1] = [8, 32, 1]

    ; A thread block offset
    v_and_b32 v[v_iak0], v[v_tid], 7
    v_lshrrev_b32 v[v_im], 3, v[v_tid]
    v_lshlrev_b32 v[v_tmp], 4, v[v_iak0]
    v_mad_u32_u24 v[v_offset_a], v[v_im], s[s_lda], v[v_tmp]
    ; A grid offset
    s_mul_i32 s[s_tmp], s[s_m_idx], s[s_lda]
    s_lshl_b32 s[s_tmp + 1], s[s_k_idx], 1
    s_add_i32 s[s_tmp], s[s_tmp], s[s_tmp + 1]
    s_add_u32  s[s_ptr_a], s[s_ptr_a], s[s_tmp]
    s_addc_u32 s[s_ptr_a + 1], s[s_ptr_a + 1], 0
    ; prefetch load A
    s_mul_i32 s[s_ptr_a + 2], s[s_m], s[s_lda]
    s_sub_i32 s[s_ptr_a + 2], s[s_ptr_a + 2], s[s_tmp]

    s_mov_b32 s[s_bs_a], 128

    buffer_load_dwordx4 v[v_gld_a0 + 0 : v_gld_a0 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]

    ; load B matrix
    ; B:
    ; thread vec: [bk0, n, bk1] = [2, 1, 16]
    ; block vec:  [bk0, n, bk1] = [2, 128, 1]
    ; B thread block offset
    v_mov_b32 v[v_tmp], 127
    v_and_b32 v[v_in], v[v_tid], v[v_tmp]
    v_lshrrev_b32 v[v_ibk0], 7, v[v_tid]
    v_lshlrev_b32 v[v_tmp], 4, v[v_in]
    ; k0 offset = ldb
    v_mad_u32_u24 v[v_offset_b], v[v_ibk0], s[s_ldb], v[v_tmp]
    ; B grid offset
    s_lshr_b32 s[s_tmp + 1], s[s_ldb], 4
    s_lshl_b32 s[s_tmp], s[s_n_idx], 4
    s_mul_i32 s[s_tmp + 2], s[s_tmp + 1], s[s_k_idx]
    s_add_u32 s[s_tmp], s[s_tmp], s[s_tmp + 2]
    s_add_u32  s[s_ptr_b], s[s_ptr_b], s[s_tmp]
    s_addc_u32 s[s_ptr_b + 1], s[s_ptr_b + 1], 0
    ; prefetch load B
    s_mul_i32 s[s_ptr_b + 2], s[s_k], s[s_tmp + 1]
    s_sub_i32 s[s_ptr_b + 2], s[s_ptr_b + 2], s[s_tmp]
    s_lshl_b32 s[s_bs_b], s[s_ldb], 2

    s_mul_i32 s[s_offset_b + 0], s[s_ldb], 2

    buffer_load_dwordx4 v[v_gld_b0 + 0 : v_gld_b0 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    buffer_load_dwordx4 v[v_gld_b0 + 4 : v_gld_b0 + 7], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], s[s_offset_b + 0] offen offset:0
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
    s_lshl_b32 s[s_wave_im], s[s_wave_im], 5
    s_lshl_b32 s[s_wave_in], s[s_wave_in], 5

    ; lane id
    v_and_b32 v[v_lane_id], 63, v[v_tid]
    v_and_b32 v[v_lane_in], 31, v[v_tid] 
    v_lshrrev_b32 v[v_lane_im], 5, v[v_lane_id]
    v_lshlrev_b32 v[v_lane_im], 2, v[v_lane_im]

    ; sst offset C
    ; m_offset = (wave_im + lane_im) * block_n
    ; n_offset = wave_n + lane_in
    ; sst_c_offset = m_offset + n_offset
    v_add_lshl_u32 v[v_sst_offset_c], v[v_lane_im], s[s_wave_im], 7
    v_add_u32 v[v_tmp], v[v_lane_in], s[s_wave_in]
    v_add_lshl_u32 v[v_sst_offset_c], v[v_tmp], v[v_sst_offset_c], 1

    ; sld/gst offset C
    ; c_in = tid % (block_n / vec_c_n)
    ; c_im = tid / (block_n / vec_c_n)
    ; sld_c_offset = c_in * vec_c_n + c_im * block_n
    ; gst_c_offset = c_in * vec_c_n + c_im * ldc
    v_and_b32 v[v_c_in], 15, v[v_tid]
    v_lshrrev_b32 v[v_c_im], 4, v[v_tid]
    v_lshlrev_b32 v[v_tmp], 4, v[v_c_in]
    v_lshl_add_u32 v[v_sld_offset_c], v[v_c_im], 8, v[v_tmp]
    v_mul_lo_u32 v[v_tmp + 1], v[v_c_im], s[s_ldc]
    v_add_u32 v[v_gst_offset_c], v[v_tmp + 1], v[v_tmp]

    ; c grid pointer
    s_mul_i32 s[s_tmp], s[s_m_idx], s[s_ldc]
    s_lshl_b32 s[s_tmp + 2], s[s_n_idx], 1
    s_add_u32 s[s_tmp + 1], s[s_tmp + 2], s[s_tmp]
    s_mul_i32 s[s_tmp], s[s_m], s[s_ldc]
    s_mul_i32 s[s_tmp], s[s_tmp], s[s_bz]
    s_add_u32 s[s_ptr_c], s[s_ptr_c], s[s_tmp + 1]
    s_addc_u32 s[s_ptr_c + 1], s[s_ptr_c + 1], 0
    s_add_u32 s[s_ptr_c], s[s_ptr_c], s[s_tmp]
    s_addc_u32 s[s_ptr_c + 1], s[s_ptr_c + 1], 0
    s_mul_i32 s[s_ptr_c + 2], s[s_m], s[s_ldc]
    s_sub_i32 s[s_ptr_c + 2], s[s_ptr_c + 2], s[s_tmp + 1]
    ; c n flag
    v_lshl_add_u32 v[v_tmp], v[v_c_in], 3, s[s_n_idx]
    v_cmp_gt_u32 vcc, s[s_n], v[v_c_in]
    v_cndmask_b32 v[v_c_n_flag],  0, 1, vcc

    ; store A to shared mem offset
    ; sst_iak0 = iak0 * (block_m + pad) * ak1
    ; sst_offset_a = sst_iak0 + v_im * 8
    v_mov_b32 v[v_tmp + 1], 512
    v_lshrrev_b32 v[v_tmp + 2], 0, v[v_im]
    v_xor_b32 v[v_tmp + 2], v[v_iak0], v[v_tmp + 2]
    v_lshlrev_b32 v[v_tmp + 2], 0, v[v_tmp + 2]
    v_and_b32 v[v_tmp + 3], 0, v[v_im]
    v_add_u32 v[v_tmp + 2], v[v_tmp + 2], v[v_tmp + 3]
    v_lshlrev_b32 v[v_tmp], 4, v[v_tmp + 2]
    v_mad_u32_u24 v[v_sst_offset_a0], v[v_iak0], v[v_tmp + 1], v[v_tmp]
    v_mov_b32 v[v_tmp], 20480
    v_add_u32 v[v_sst_offset_a1], v[v_tmp], v[v_sst_offset_a0]

    ; store B to shared mem offset. when B is stored to shared mem, B datatype is bf16/fp16
    ; bk1 = max(ak1, bk1_gld, 8)
    ; sst_in = v_in * bk1 * n1 = v_in * 8 * 1
    ; sst_ibk0 = v_ibk0 * block_n * bk1_gld = v_ibk0 * 128 * 16
    ; sst_offset_b = sst_in + sst_ibk0
    ; padding = sst_offset_b / 64 * 8
    ; sst_offset_b = sst_offset_b + padding
    v_lshlrev_b32 v[v_tmp], 3, v[v_in]
    v_lshlrev_b32 v[v_tmp + 1], 11, v[v_ibk0]
    v_add_u32 v[v_sst_offset_b0], v[v_tmp], v[v_tmp + 1]
    v_lshlrev_b32 v[v_sst_offset_b0], 1, v[v_sst_offset_b0]
    v_mov_b32 v[v_tmp], 4096
    v_add_u32 v[v_sst_offset_b0], v[v_sst_offset_b0], v[v_tmp]
    v_mov_b32 v[v_tmp], 20480
    v_add_u32 v[v_sst_offset_b1], v[v_tmp], v[v_sst_offset_b0]

    ; load A to shared mem offset
    ; sld_iak0 = laneid / inst_m * ((block_m + pad) * ak1)
    ; sld_im = lane_id % inst_m + wave_im
    ; sld_offset_a = sld_im * ak1 + sld_iak0
    v_lshrrev_b32 v[v_sld_iak0], 5, v[v_lane_id]
    v_and_b32 v[v_sld_im], 31, v[v_lane_id]
    v_lshrrev_b32 v[v_tmp + 4], 0, v[v_sld_im]
    v_and_b32 v[v_tmp + 5], 0, v[v_sld_im]
    v_xor_b32 v[v_tmp], v[v_tmp + 4], v[v_sld_iak0]
    v_lshlrev_b32 v[v_tmp], 0, v[v_tmp]
    v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp + 5]
    
    v_mov_b32 v[v_tmp + 1], 2
    v_add_u32 v[v_tmp + 1], v[v_tmp + 1], v[v_sld_iak0]
    v_xor_b32 v[v_tmp + 1], v[v_tmp + 4], v[v_tmp + 1]
    v_lshlrev_b32 v[v_tmp + 1], 0, v[v_tmp + 1]
    v_add_u32 v[v_tmp + 1], v[v_tmp + 1], v[v_tmp + 5]

    v_mov_b32 v[v_tmp + 2], 4
    v_add_u32 v[v_tmp + 2], v[v_tmp + 2], v[v_sld_iak0]
    v_xor_b32 v[v_tmp + 2], v[v_tmp + 4], v[v_tmp + 2]
    v_lshlrev_b32 v[v_tmp + 2], 0, v[v_tmp + 2]
    v_add_u32 v[v_tmp + 2], v[v_tmp + 2], v[v_tmp + 5]

    v_mov_b32 v[v_tmp + 3], 6
    v_add_u32 v[v_tmp + 3], v[v_tmp + 3], v[v_sld_iak0]
    v_xor_b32 v[v_tmp + 3], v[v_tmp + 4], v[v_tmp + 3]
    v_lshlrev_b32 v[v_tmp + 3], 0, v[v_tmp + 3]
    v_add_u32 v[v_tmp + 3], v[v_tmp + 3], v[v_tmp + 5]

    v_mov_b32 v[v_tmp + 4], 256
    v_mul_lo_u32 v[v_sld_iak0], v[v_tmp + 4], v[v_sld_iak0] 
    v_add_lshl_u32 v[v_sld_im], v[v_tmp], s[s_wave_im], 3
    v_add_lshl_u32 v[v_sld_offset_a0], v[v_sld_iak0], v[v_sld_im], 1
    
    v_add_lshl_u32 v[v_sld_im], v[v_tmp + 1], s[s_wave_im], 3
    v_add_lshl_u32 v[v_sld_offset_a0 + 1], v[v_sld_iak0], v[v_sld_im], 1

    v_add_lshl_u32 v[v_sld_im], v[v_tmp + 2], s[s_wave_im], 3
    v_add_lshl_u32 v[v_sld_offset_a0 + 2], v[v_sld_iak0], v[v_sld_im], 1

    v_add_lshl_u32 v[v_sld_im], v[v_tmp + 3], s[s_wave_im], 3
    v_add_lshl_u32 v[v_sld_offset_a0 + 3], v[v_sld_iak0], v[v_sld_im], 1

    v_mov_b32 v[v_tmp], 20480
    v_add_u32 v[v_sld_offset_a1], v[v_tmp], v[v_sld_offset_a0]
    
    v_add_u32 v[v_sld_offset_a1 + 1], v[v_tmp], v[v_sld_offset_a0 + 1]

    v_add_u32 v[v_sld_offset_a1 + 2], v[v_tmp], v[v_sld_offset_a0 + 2]

    v_add_u32 v[v_sld_offset_a1 + 3], v[v_tmp], v[v_sld_offset_a0 + 3]


    ; load B to shared mem offset
    ; k1 = max(ak1, bk1)
    ; sld_ibk0 = laneid / inst_n * (block_n * k1)
    ; sld_in = laneid % inst_n + wave_in
    ; sld_offset_b = sld_ibk0 + sld_in * bk1
    ; padding = sld_offset_b / 64 * 8
    ; sld_offset_b = padding + sld_offset_b
    v_lshrrev_b32 v[v_sld_ibk0], 5, v[v_lane_id]
    v_lshlrev_b32 v[v_sld_ibk0], 10, v[v_sld_ibk0]
    v_and_b32 v[v_sld_in], 31, v[v_lane_id]
    v_add_lshl_u32 v[v_sld_in], v[v_sld_in], s[s_wave_in], 3
    v_add_u32 v[v_sld_offset_b0], v[v_sld_in], v[v_sld_ibk0]
    v_lshlrev_b32 v[v_sld_offset_b0], 1, v[v_sld_offset_b0]
    v_mov_b32 v[v_tmp], 4096
    v_add_u32 v[v_sld_offset_b0], v[v_sld_offset_b0], v[v_tmp]
    v_mov_b32 v[v_tmp], 20480
    v_add_u32 v[v_sld_offset_b1], v[v_tmp], v[v_sld_offset_b0]

    ; duplicate scale 
    s_waitcnt vmcnt(3)
    v_mov_b32 v[v_scale + 1], v[v_scale + 0]

    ; v_pk_mul_f32 v[v_sub_magic_num + 0 : v_sub_magic_num + 1], v[v_scale + 0 : v_scale + 1], v[v_sub_magic_num + 0 : v_sub_magic_num + 1]
    ;v_mul_f32 v[v_sub_magic_num + 0], v[v_scale + 0], v[v_sub_magic_num + 0]
    ;v_mul_f32 v[v_sub_magic_num + 1], v[v_scale + 1], v[v_sub_magic_num + 1]

    ; clear ACC vgpr
    .cnt = 0
    .rept 16
        v_mov_b32 v[v_c + .cnt], 0
        .cnt = .cnt + 1
    .endr

    buffer_load_dwordx4 v[v_gld_a1 + 0 : v_gld_a1 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0
    buffer_load_dwordx4 v[v_gld_b1 + 0 : v_gld_b1 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    buffer_load_dwordx4 v[v_gld_b1 + 4 : v_gld_b1 + 7], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], s[s_offset_b] offen offset:0
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]

    ; store gld_a0 to lds
    s_waitcnt vmcnt(5)
    ds_write_b128 v[v_sst_offset_a0], v[v_gld_a0 : v_gld_a0 + 3], offset: 0
    buffer_load_dwordx4 v[v_gld_a0 + 0 : v_gld_a0 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0

    ; dequant gld_b0
    s_waitcnt vmcnt(5)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 0

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 2, v_sel_b + 0, v_sub_magic_num, v_scale
    buffer_load_dwordx4 v[v_gld_b0 + 0 : v_gld_b0 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 1

    s_waitcnt vmcnt(5)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 4, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 2 + 128 * 8 * 2 * 0

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 6, v_sel_b + 0, v_sub_magic_num, v_scale
    buffer_load_dwordx4 v[v_gld_b0 + 4 : v_gld_b0 + 7], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], s[s_offset_b] offen offset:0
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 2 + 128 * 8 * 2 * 1

    s_waitcnt lgkmcnt(0)
    s_barrier
   
    s_mov_b32 s[s_kitr], 64 * (1 + 0) ; 1 prefetch
    s_cmp_le_u32 s[s_k_per_cta], s[s_kitr]
    s_cbranch_scc1 label_gemm_rrr_loop_last_1

    s_mov_b32 s[s_kitr], 64 * (1 + 1)
    s_cmp_le_u32 s[s_k_per_cta], s[s_kitr]
    s_cbranch_scc1 label_gemm_rrr_loop_last_2

    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a0], offset: 0
    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b0], offset: 0 
label_gemm_rrr_loop_begin:
    ; load from lds and do mfma
    ;.mfma_wg1x1_w1x4_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c
    ; s_barrier
    ds_read_b128 v[v_sld_b1 + 0 : v_sld_b1 + 3], v[v_sld_offset_b0], offset: 128 * 8 * 2 * 2 * 1
    ds_read_b128 v[v_sld_a1 + 0 : v_sld_a1 + 3], v[v_sld_offset_a0 + 1], offset: (32 + 0) * 8 * 2 * 2 * 1
    ; global load n + 2
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]
    
    s_waitcnt lgkmcnt(2)

    s_waitcnt vmcnt(5)
    ds_write_b128 v[v_sst_offset_a1], v[v_gld_a1 : v_gld_a1 + 3], offset: 0
    buffer_load_dwordx4 v[v_gld_a1 + 0 : v_gld_a1 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 0 : v_c + 15]
    
    s_waitcnt vmcnt(5)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b1], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 0 : v_c + 15]
    
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 2, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b1], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 1

    s_waitcnt lgkmcnt(3)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 0 : v_c + 15]
    
    buffer_load_dwordx4 v[v_gld_b1 + 0 : v_gld_b1 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a0 + 2], offset: (32 + 0) * 8 * 2 * 2 * 2
    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b0], offset: 128 * 8 * 2 * 2 * 2

    s_waitcnt vmcnt(5)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 4, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b1], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 2 + 128 * 8 * 2 * 0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 0 : v_c + 15]
    
    ds_read_b128 v[v_sld_b1 + 0 : v_sld_b1 + 3], v[v_sld_offset_b0], offset: 128 * 8 * 2 * 2 * 3
    ds_read_b128 v[v_sld_a1 + 0 : v_sld_a1 + 3], v[v_sld_offset_a0 + 3], offset: (32 + 0) * 8 * 2 * 2 * 3
    s_waitcnt lgkmcnt(3)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 0 : v_c + 15]

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 6, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b1], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 2 + 128 * 8 * 2 * 1

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 0 : v_c + 15]
    
    buffer_load_dwordx4 v[v_gld_b1 + 4 : v_gld_b1 + 7], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], s[s_offset_b] offen offset:0
    s_waitcnt lgkmcnt(0)
    s_barrier

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 0 : v_c + 15]

    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]

    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a1], offset: 0
    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b1], offset: 0 

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 0 : v_c + 15]
    
    ; load from lds and do mfma
    ds_read_b128 v[v_sld_b1 + 0 : v_sld_b1 + 3], v[v_sld_offset_b1], offset: 128 * 8 * 2 * 2 * 1
    ds_read_b128 v[v_sld_a1 + 0 : v_sld_a1 + 3], v[v_sld_offset_a1 + 1], offset: (32 + 0) * 8 * 2 * 2 * 1
    s_waitcnt lgkmcnt(2)

    s_waitcnt vmcnt(5)
    ds_write_b128 v[v_sst_offset_a0], v[v_gld_a0 : v_gld_a0 + 3], offset: 0
    buffer_load_dwordx4 v[v_gld_a0 + 0 : v_gld_a0 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 0 : v_c + 15]

    s_waitcnt vmcnt(5)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 0 : v_c + 15]

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 2, v_sel_b + 0, v_sub_magic_num, v_scale
    buffer_load_dwordx4 v[v_gld_b0 + 0 : v_gld_b0 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 1

    s_waitcnt lgkmcnt(3)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 0 : v_c + 15]
    
    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a1 + 2], offset: (32 + 0) * 8 * 2 * 2 * 2
    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b1], offset: 128 * 8 * 2 * 2 * 2
    
    s_waitcnt vmcnt(5)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 4, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 2 + 128 * 8 * 2 * 0

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 0 : v_c + 15]
    
    ds_read_b128 v[v_sld_b1 + 0 : v_sld_b1 + 3], v[v_sld_offset_b1], offset: 128 * 8 * 2 * 2 * 3
    ds_read_b128 v[v_sld_a1 + 0 : v_sld_a1 + 3], v[v_sld_offset_a1 + 3], offset: (32 + 0) * 8 * 2 * 2 * 3
    s_waitcnt lgkmcnt(3)

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 0 : v_sld_a0 + 1], v[v_sld_b0 + 0 : v_sld_b0 + 1], v[v_c + 0 : v_c + 15]
    
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b0 + 6, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 2 + 128 * 8 * 2 * 1

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a0 + 2 : v_sld_a0 + 3], v[v_sld_b0 + 2 : v_sld_b0 + 3], v[v_c + 0 : v_c + 15]

    buffer_load_dwordx4 v[v_gld_b0 + 4 : v_gld_b0 + 7], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], s[s_offset_b] offen offset:0
    s_waitcnt lgkmcnt(0)
    s_barrier

    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a1 + 0 : v_sld_a1 + 1], v[v_sld_b1 + 0 : v_sld_b1 + 1], v[v_c + 0 : v_c + 15]
    
    ds_read_b128 v[v_sld_a0 + 0 : v_sld_a0 + 3], v[v_sld_offset_a0], offset: 0
    ds_read_b128 v[v_sld_b0 + 0 : v_sld_b0 + 3], v[v_sld_offset_b0], offset: 0 
    
    v_mfma_f32_32x32x8bf16_1k v[v_c + 0 : v_c + 15], v[v_sld_a1 + 2 : v_sld_a1 + 3], v[v_sld_b1 + 2 : v_sld_b1 + 3], v[v_c + 0 : v_c + 15]
 
    ; store gld_a0 to lds
    ; dequant gld_b0
   
    s_add_u32 s[s_kitr], 128, s[s_kitr] ; 64 * (1 + 1) 1 prefetch
    s_cmp_lt_u32 s[s_kitr], s[s_k_per_cta]
    s_cbranch_scc1 label_gemm_rrr_loop_begin

    s_sub_u32 s[s_kitr], s[s_kitr], 64
    s_cmp_lt_u32 s[s_kitr], s[s_k_per_cta]
    s_cbranch_scc1 label_gemm_rrr_loop_last_2
    
    s_branch label_gemm_rrr_loop_last_1

label_gemm_rrr_loop_last_2:
    ; load from lds and do mfma
    .mfma_wg1x1_w1x4_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c
    s_barrier
    
    ; store gld_a1 to lds
    s_waitcnt vmcnt(2)
    ds_write_b128 v[v_sst_offset_a0], v[v_gld_a1 : v_gld_a1 + 3], offset: 0

    ; dequant gld_b0
    s_waitcnt vmcnt(1)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 0, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 0

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 2, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 0 + 128 * 8 * 2 * 1

    s_waitcnt vmcnt(0)
    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 4, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 2 + 128 * 8 * 2 * 0

    .dequant_int8_1x8 v_tmp, v_fp32_base, v_gld_b1 + 6, v_sel_b + 0, v_sub_magic_num, v_scale
    ds_write_b128 v[v_sst_offset_b0], v[v_tmp : v_tmp + 3], offset: 128 * 16 * 2 * 2 + 128 * 8 * 2 * 1

    s_waitcnt lgkmcnt(0)
    s_barrier

    ; load from lds and do mfma
    .mfma_wg1x1_w1x4_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c
    s_barrier

    s_branch label_write_out_c 
    
label_gemm_rrr_loop_last_1:
    ; load from lds and do mfma
    .mfma_wg1x1_w1x4_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a0, v_sld_offset_b0, v_c


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
    .rept 4
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + 0 + .v_c_inst_cnt * 4 + 0], offset: 0 + 2048 * .v_c_inst_cnt + 256 * 0
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + 0 + .v_c_inst_cnt * 4 + 1], offset: 0 + 2048 * .v_c_inst_cnt + 256 * 1
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + 0 + .v_c_inst_cnt * 4 + 2], offset: 0 + 2048 * .v_c_inst_cnt + 256 * 2
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + 0 + .v_c_inst_cnt * 4 + 3], offset: 0 + 2048 * .v_c_inst_cnt + 256 * 3
        .v_c_inst_cnt = .v_c_inst_cnt + 1
    .endr

    s_waitcnt lgkmcnt(0)
    s_barrier

    ; load from lds
    ; imm_offset = 16 * threadim.x * i

    ds_read_b128 v[v_c + 0 + 0 : v_c + 0 + 3], v[v_sld_offset_c], offset: 16 * 256 * 0

    ds_read_b128 v[v_c + 4 + 0 : v_c + 4 + 3], v[v_sld_offset_c], offset: 16 * 256 * 1

    s_waitcnt lgkmcnt(0)
    s_barrier

    ; store res to global
    v_cmpx_eq_u32 vcc, 1, v[v_c_n_flag]    

    s_mul_i32 s[s_tmp], 0, s[s_ldc]
    buffer_store_dwordx4 v[v_c + 0 + 0 : v_c + 0 + 3], v[v_gst_offset_c], s[s_ptr_c + 0 : s_ptr_c + 3], s[s_tmp] offen offset: 0

    s_mul_i32 s[s_tmp], 16, s[s_ldc]
    buffer_store_dwordx4 v[v_c + 4 + 0 : v_c + 4 + 3], v[v_gst_offset_c], s[s_ptr_c + 0 : s_ptr_c + 3], s[s_tmp] offen offset: 0

    s_mov_b64 exec, -1

l_local_compute_signal:
    ; find proper flag
    v_mov_b32 v[v_imm], 1
    s_lshr_b32 s[s_offset_local_flag], s[s_bx], 2
    s_lshl_b32 s[s_offset_local_flag], s[s_offset_local_flag], 2
    v_mov_b32 v[v_offset_flag], 0
    v_cmpx_gt_u32 v[v_imm], v[v_tid]
    global_atomic_add v[v_flag], v[v_offset_flag], v[v_imm], s[s_local_flag : s_local_flag + 1] glc
    s_add_u32 s[s_flag_checker], s[s_m], 31
    s_lshr_b32 s[s_flag_checker], s[s_flag_checker], 5
    s_lshl_b32 s[s_flag_checker], s[s_flag_checker], 2
    s_waitcnt vmcnt(0)
    v_readfirstlane_b32 s[s_flag], v[v_flag]
    s_cmp_eq_u32 s[s_flag], s[s_flag_checker]
    s_cbranch_scc0 l_end_bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk

    ; begin multicard barrier
    s_mov_b64 exec -1
    v_mov_b32 v[v_imm], 4
    v_cmpx_gt_u32 v[v_imm], v[v_tid]
    v_lshlrev_b32 v[v_barrier_offset], 3, v[v_tid]
    v_lshlrev_b32 v[v_local_barrier_offset], 2, v[v_tid]
    global_load_dwordx2 v[v_barrier_addr : v_barrier_addr + 1], v[v_barrier_offset], s[s_ka : s_ka + 1] offset:0+k_world_barrier 
    s_lshl_b64 s[s_local_rank : s_local_rank + 1], s[s_local_rank : s_local_rank + 1], 2
    v_mov_b32 v[v_local_rank], s[s_local_rank + 1]
    v_mov_b32 v[v_barrier_flag], s[s_multigpu_barrier_flag]
    s_waitcnt vmcnt(0)
    v_add_co_u32_e32 v[v_barrier_addr], vcc, s[s_local_rank], v[v_barrier_addr]
    v_addc_co_u32_e32 v[v_barrier_addr + 1], vcc, v[v_local_rank], v[v_barrier_addr + 1], vcc
    global_store_dword v[v_barrier_addr : v_barrier_addr + 1], v[v_barrier_flag], off

l_begin_barrier_check:
    global_load_dword v[v_barrier_flag_check], v[v_local_barrier_offset], s[s_local_barrier : s_local_barrier + 1] glc
    s_waitcnt vmcnt(0) lgkmcnt(0)
    ; .print v_flag, s_print, s_bx, v_tid, v_tmp + 7
    v_cmp_le_u32 vcc, s[s_multigpu_barrier_flag], v[v_barrier_flag_check]
    s_andn2_b64 exec, exec, vcc
    s_cbranch_execnz l_begin_barrier_check
    
    s_mov_b64 exec -1

l_end_barrier_check:
    v_mov_b32 v[v_imm], 4
    v_cmpx_gt_u32 v[v_imm], v[v_tid]
    v_mov_b32 v[v_imm], 0
    global_store_dword v[v_imm], v[v_local_barrier_offset], s[s_local_barrier : s_local_barrier + 1] glc
    global_store_dword v[v_offset_flag], v[v_offset_flag], s[s_local_flag : s_local_flag + 1] glc
    

l_end_bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk: 
    ; .print v_offset_a, s_print, s_bx, v_tid, v_tmp + 7
    s_mov_b64 exec -1
    s_endpgm
.rodata
.p2align 6
.amdhsa_kernel bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk
    .amdhsa_group_segment_fixed_size 40960
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_sgpr_workgroup_id_z 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 116
    .amdhsa_next_free_sgpr 88
    .amdhsa_ieee_mode 0
    .amdhsa_dx10_clamp 0
    .amdhsa_accum_offset 116
    # .amdhsa_wavefront_size32 1
    # .amdhsa_workgroup_processor_mode 0
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [ 1, 0 ]
amdhsa.kernels:
  - .name: bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk
    .symbol: bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipeline_interleaved_splitk.kd
    .sgpr_count: 88
    .vgpr_count: 116
    .kernarg_segment_align: 8
    .kernarg_segment_size: 256 
    .group_segment_fixed_size: 40960
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size: [256, 1, 1]
    .max_flat_workgroup_size: 256
    .args: 
      - { .name k_ptr_c, .size: 8, .offset: 0, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: false} 
      - { .name k_ptr_a, .size: 8, .offset: 8, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: true} 
      - { .name k_ptr_b, .size: 8, .offset: 16, .value_kind: global_buffer, .value_type: f16, .address_space: global, .is_const: true} 
      - { .name k_ptr_scale, .size: 8, .offset: 24, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: true} 
      - { .name k_m, .size: 4, .offset: 32, .value_kind: by_value, .value_type: i32} 
      - { .name k_n, .size: 4, .offset: 36, .value_kind: by_value, .value_type: i32} 
      - { .name k_k, .size: 4, .offset: 40, .value_kind: by_value, .value_type: i32} 
      - { .name k_lda, .size: 4, .offset: 44, .value_kind: by_value, .value_type: i32} 
      - { .name k_ldb, .size: 4, .offset: 48, .value_kind: by_value, .value_type: i32} 
      - { .name k_ldc, .size: 4, .offset: 52, .value_kind: by_value, .value_type: i32} 
      - { .name k_k_per_cta, .size: 4, .offset: 56, .value_kind: by_value, .value_type: i32} 
      - { .name k_print, .size: 8, .offset: 60, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_multigpu_barrier_flag, .size: 4, .offset: 68, .value_kind: by_value, .value_type: i32} 
      - { .name k_local_flag, .size: 8, .offset: 72, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_world_barrier, .size: 8, .offset: 80, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_world_barrier1, .size: 8, .offset: 88, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_world_barrier2, .size: 8, .offset: 96, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_world_barrier3, .size: 8, .offset: 104, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_world_barrier4, .size: 8, .offset: 112, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_world_barrier5, .size: 8, .offset: 120, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_world_barrier6, .size: 8, .offset: 128, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_world_barrier7, .size: 8, .offset: 136, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_local_out, .size: 8, .offset: 144, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_peer_comm_buffer, .size: 8, .offset: 152, .value_kind: global_buffer, .value_type: f32, .address_space: global, .is_const: false} 
      - { .name k_local_rank, .size: 8, .offset: 160, .value_kind: by_value, .value_type: i64} 

...
.end_amdgpu_metadata
