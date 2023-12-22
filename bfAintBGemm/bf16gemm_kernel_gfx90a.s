
.macro .print v_val, s_out, s_bx, v_tid, v_offset
    ;s_mov_b64 exec, -1
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
.set s_tmp,             42
.set s_end,             51

;vgpr
.set v_c,               0
.set v_a0,              16
.set v_b0,              20
.set v_a1,              8
.set v_b1,              12
.set v_p0,              16
.set v_q0,              24
.set v_smem_store,      32
.set v_smem_load_a,     33
.set v_smem_load_b,     34
.set v_smem_store_c,    35
.set v_smem_load_c,     36
.set v_scale,           37
.set v_lane_lo,         38
.set v_lane_hi,         39
.set v_offset_a_k0,     40
.set v_offset_a,        41
.set v_offset_b_k0,     42
.set v_offset_b,        43
.set v_offset_c,        44
.set v_wave_p,          45
.set v_wave_q,          46
.set v_lane_col,        47
.set v_lane_row,        48
.set v_tmp,             49
.set v_tid,             63

.text
.global bf16gemm_rrr
.p2align 8
.type bf16gemm_rrr,@function
bf16gemm_rrr:
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

    s_waitcnt lgkmcnt(0)


    ; A and C matrix is bf16 datatype
    s_lshl_b32 s[s_lda], s[s_lda], 1
    s_lshl_b32 s[s_ldc], s[s_ldc], 1

    ; thread block mapping
    ; m block id: bid x
    ; n block id: bid y
    s_lshl_b32 s[s_m_idx], s[s_bx], 5
    s_lshl_b32 s[s_n_idx], s[s_by], 6
    
    ; load A/B matrix
    ; A:
    ; thread vec: [k0, m, k1] = [ 1,  2,  8]
    ; block vec:  [k0, m, k1] = [ 8, 16,  1]
    ; B:
    ; thread vec: [k0, n, k1] = [ 1,  8,  4]
    ; block vec:  [k0, n, k1] = [16,  8,  1]

    ; A thread block offset
    v_and_b32 v[v_offset_a_k0], v[v_tid], 7
    v_lshrrev_b32 v[v_tmp], 3, v[v_tid]
    v_lshlrev_b32 v[v_offset_a_k0], 4, v[v_offset_a_k0]
    v_mad_u32_u24 v[v_offset_a], v[v_tmp], s[s_lda], v[v_offset_a_k0]
    ; A grid offset
    s_mul_i32 s[s_tmp], s[s_m_idx], s[s_lda]
    s_add_u32  s[s_ptr_a], s[s_ptr_a], s[s_tmp]
    s_addc_u32 s[s_ptr_a + 1], s[s_ptr_a + 1], 0
    s_lshl_b32 s[s_offset_a], s[s_lda], 4
    ; prefetch load A
    s_mul_i32 s[s_ptr_a + 2], s[s_m], s[s_lda]
    
    buffer_load_dwordx4 v[v_p0 + 0 : v_p0 + 3], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], 0 offen offset:0
    buffer_load_dwordx4 v[v_p0 + 4 : v_p0 + 7], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], s[s_offset_a] offen offset:0
    s_mov_b32 s[s_bs_a], 128

    ; B thread block offset
    v_and_b32 v[v_tmp], v[v_tid], 7
    v_lshrrev_b32 v[v_offset_b_k0], 3, v[v_tid]
    v_lshlrev_b32 v[v_tmp], 3, v[v_tmp]
    ; k0 offset = t_k1 * ldb
    s_lshl_b32 s[s_tmp], s[s_ldb], 2
    v_mad_u32_u24 v[v_offset_b], v[v_offset_b_k0], s[s_tmp], v[v_tmp]
    ; B grid offset
    s_add_u32  s[s_ptr_b], s[s_ptr_b], s[s_n_idx]
    s_addc_u32 s[s_ptr_b + 1], s[s_ptr_b + 1], 0
    s_lshl_b32 s[s_bs_b], s[s_ldb], 6
    s_mov_b32 s[s_offset_b], s[s_ldb]
    s_mul_i32 s[s_offset_b + 1], s[s_ldb], 2
    s_mul_i32 s[s_offset_b + 2], s[s_ldb], 3
    ; prefetch load B
    s_mul_i32 s[s_ptr_b + 2], s[s_k], s[s_ldb]

    buffer_load_dwordx2 v[v_q0 + 0 : v_q0 + 1], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], 0 offen offset:0
    buffer_load_dwordx2 v[v_q0 + 2 : v_q0 + 3], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], s[s_offset_b] offen offset:0
    buffer_load_dwordx2 v[v_q0 + 4 : v_q0 + 5], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], s[s_offset_b + 1] offen offset:0
    buffer_load_dwordx2 v[v_q0 + 6 : v_q0 + 7], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], s[s_offset_b + 2] offen offset:0

    ; load scale
    ; Scale:
    ; thread vec: [n]         = [ 1]
    ; block vec:  [k0, n, k1] = [16,  8,  1]
    v_and_b32 v[v_tmp], v[v_tid], 7
    v_lshlrev_b32 v[v_tmp], 2, v[v_tmp]
    s_lshl_b32 s[s_tmp], s[s_n_idx], 2
    s_add_u32  s[s_ptr_scale], s[s_ptr_scale], s[s_tmp]
    s_addc_u32 s[s_ptr_scale + 1], s[s_ptr_scale + 1], 0
    s_lshl_b32 s[s_ptr_scale + 2], s[s_n], 2
    buffer_load_dword v[v_scale], v[v_tmp], s[s_ptr_scale : s_ptr_scale + 3], 0 offen offset:0

    .print v_scale, s_print, s_bx, v_tid, v_tmp+4

    

    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel bf16gemm_rrr
    .amdhsa_group_segment_fixed_size 16384
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_sgpr_workgroup_id_y 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 52
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
  - .name: bf16gemm_rrr
    .symbol: bf16gemm_rrr.kd
    .sgpr_count: 52
    .vgpr_count: 64
    .kernarg_segment_align: 8
    .kernarg_segment_size: 72
    .group_segment_fixed_size: 16384
    .private_segment_fixed_size: 0
    .wavefront_size: 64
    .reqd_workgroup_size : [128, 1, 1]
    .max_flat_workgroup_size: 256
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

