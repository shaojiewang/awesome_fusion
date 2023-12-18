.text
.global bf16gemm_rrr
.p2align 8
.type bf16gemm_rrr,@function
bf16gemm_rrr:
; This is just an example, not the optimal one
.set s_karg,            0   ; kernel argument
.set s_bx,              2   ; blockIdx

.set s_ptr_in,          4
.set s_ptr_out,         6
.set s_loops_per_block, 8
.set s_stride_block,    10
.set s_tmp,             12
.set s_gdx,             16

.set v_buf,             0
.set v_offset,          16
.set v_tmp,             32

    ; http://www.hsafoundation.com/html/Content/Runtime/Topics/02_Core/hsa_kernel_dispatch_packet_t.htm
    ;s_load_dword s[s_gdx],                  s[s_dptr:s_dptr+1], 12
    ;s_waitcnt           lgkmcnt(0)
    ;s_lshr_b32      s[s_gdx],   s[s_gdx],   8
    ;s_mov_b32   s[s_gdx], 72    ; num_cu

    s_load_dwordx2 s[s_ptr_in:s_ptr_in+1],      s[s_karg:s_karg+1],     0
    s_load_dwordx2 s[s_ptr_out:s_ptr_out+1],    s[s_karg:s_karg+1],     8
    s_load_dword s[s_loops_per_block],          s[s_karg:s_karg+1],     16
    s_load_dword s[s_gdx],                      s[s_karg:s_karg+1],     20

    s_mul_i32 s[s_tmp+1], s[s_bx], 256*4    ; blockIdx*blockDim*4
    v_lshlrev_b32 v[v_tmp], 2, v0           ; threadIdx*4
    v_add_u32 v[v_offset+0], s[s_tmp+1], v[v_tmp]    ; (blockIdx*blockDim + threadIdx)*4
    v_lshlrev_b32 v[v_offset+0], 2, v[v_offset+0]    

    s_waitcnt           lgkmcnt(0)

    s_mul_i32 s[s_tmp],  s[s_gdx],  256*4*4   ; gridDim*blockDim*float4
    v_add_u32 v[v_offset+1],    s[s_tmp],   v[v_offset+0]
    v_add_u32 v[v_offset+2],    s[s_tmp],   v[v_offset+1]
    v_add_u32 v[v_offset+3],    s[s_tmp],   v[v_offset+2]

    s_mul_i32 s[s_tmp],  s[s_gdx],  256*4   ; gridDim*blockDim*4
    s_lshl_b32 s[s_stride_block],   s[s_tmp],   4   ; unroll 16, gridDim*blockDim*4*workload

label_memcopy_start:

    s_add_u32   s[s_ptr_in],   s[s_stride_block], s[s_ptr_in]
    s_addc_u32  s[s_ptr_in+1], s[s_ptr_in+1], 0

    s_waitcnt       vmcnt(0)


    s_add_u32   s[s_ptr_out],   s[s_stride_block], s[s_ptr_out]
    s_addc_u32  s[s_ptr_out+1], s[s_ptr_out+1], 0

    s_sub_u32 s[s_loops_per_block], s[s_loops_per_block], 1
    s_cmp_eq_u32 s[s_loops_per_block], 0
    s_waitcnt       vmcnt(0)
    s_cbranch_scc0  label_memcopy_start
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel bf16gemm_rrr
    .amdhsa_group_segment_fixed_size 0
    .amdhsa_user_sgpr_dispatch_ptr 0
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_system_sgpr_workgroup_id_x 1
    .amdhsa_system_vgpr_workitem_id 0
    .amdhsa_next_free_vgpr 64
    .amdhsa_next_free_sgpr 32
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
    .sgpr_count: 32
    .vgpr_count: 64
    .kernarg_segment_align: 8
    .kernarg_segment_size: 24
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .wavefront_size: 32
    .reqd_workgroup_size : [256, 1, 1]
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

