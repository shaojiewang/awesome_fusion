from dataclasses import dataclass

@dataclass
class KernelMacro:
    macro_name : str
    macro_body : str
    
@dataclass
class PrintMacro(KernelMacro):
    def __init__(self, name):
        kernel_body = self.write_macro()
        super(PrintMacro, self).__init__(name, kernel_body)

    def write_macro(self):
        PRINT = """
.macro .print v_val, s_out, s_bx, v_tid, v_offset
    ;s_mov_b64 exec, -1
    s_nop 64
    s_cmp_eq_u32 s[\\s_bx], 0
    ;s_cbranch_scc0 L_endhere
    ;v_cmpx_eq_u32 0, v0
    v_lshlrev_b32 v[\\v_offset], 3, v[\\v_tid]
    s_waitcnt lgkmcnt(0)
    s_waitcnt vmcnt(0)
    global_store_dword v[\\v_offset], v[\\v_tid], s[\\s_out:\\s_out+1], offset:0x0
    global_store_dword v[\\v_offset], v[\\v_val], s[\\s_out:\\s_out+1], offset:0x0004
    s_waitcnt vmcnt(0)
    ;s_mov_b64 exec, -1
;L_endhere:
    s_endpgm  
.endm
"""
        return PRINT

@dataclass
class DequantMacro(KernelMacro):
    def __init__(self, name):
        macro_body = self.write_macro()
        super(DequantMacro, self).__init__(name, macro_body)

    def write_macro(self):
        DEQUANT = """
.macro .dequant_int8_1x8 v_tmp, v_base, v_gld_b, v_sel_b, v_sub_magic_num, v_scale
;.endm
;.macro fake0
    v_perm_b32 v[\\v_tmp + 0], v[\\v_base], v[\\v_gld_b], v[\\v_sel_b + 0]
    v_perm_b32 v[\\v_tmp + 1], v[\\v_base], v[\\v_gld_b], v[\\v_sel_b + 1]
    v_perm_b32 v[\\v_tmp + 2], v[\\v_base], v[\\v_gld_b], v[\\v_sel_b + 2]
    v_perm_b32 v[\\v_tmp + 3], v[\\v_base], v[\\v_gld_b], v[\\v_sel_b + 3]

    v_pk_add_f32 v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    v_pk_add_f32 v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]

    v_pk_mul_f32 v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_tmp + 0 : v_tmp + 1]
    v_pk_mul_f32 v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_tmp + 2 : v_tmp + 3]

    ;v_pk_fma_f32 v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    ;v_pk_fma_f32 v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]

    v_pack_b32_f16 v[\\v_tmp + 0], v[\\v_tmp + 0], v[\\v_tmp + 1], op_sel: [1, 1]
    v_pack_b32_f16 v[\\v_tmp + 1], v[\\v_tmp + 2], v[\\v_tmp + 3], op_sel: [1, 1]

    v_perm_b32 v[\\v_tmp + 4], v[\\v_base], v[\\v_gld_b + 1], v[\\v_sel_b + 0]
    v_perm_b32 v[\\v_tmp + 5], v[\\v_base], v[\\v_gld_b + 1], v[\\v_sel_b + 1]
    v_perm_b32 v[\\v_tmp + 6], v[\\v_base], v[\\v_gld_b + 1], v[\\v_sel_b + 2]
    v_perm_b32 v[\\v_tmp + 7], v[\\v_base], v[\\v_gld_b + 1], v[\\v_sel_b + 3]

    v_pk_add_f32 v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    v_pk_add_f32 v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]

    v_pk_mul_f32 v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_tmp + 4 : v_tmp + 5]
    v_pk_mul_f32 v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_tmp + 6 : v_tmp + 7]

    ;v_pk_fma_f32 v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    ;v_pk_fma_f32 v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    
    v_pack_b32_f16 v[\\v_tmp + 2], v[\\v_tmp + 4], v[\\v_tmp + 5], op_sel: [1, 1]
    v_pack_b32_f16 v[\\v_tmp + 3], v[\\v_tmp + 6], v[\\v_tmp + 7], op_sel: [1, 1]

.endm
"""
        return DEQUANT

@dataclass
class MfmaMacro(KernelMacro):
    def __init__(self, name, repeat_m, repeat_n):
        mfma_body = ""
        if repeat_m == 1 and repeat_n == 1:
            mfma_body = self.write_wg1x1_macro()
        if repeat_m == 2 and repeat_n == 2:
            mfma_body = self.write_wg2x2_macro()
        super(MfmaMacro, self).__init__(name, mfma_body)

    def write_wg1x1_macro(self):
        MFMA = """
.macro .mfma_wg1x1_w1x4_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a, v_sld_offset_b, v_c
;    .rept 8
;        v_fmac_f32 v0, v1, v2
;    .endr
;.endm
;.macro fake1
    ds_read_b128 v[\\v_sld_a0 + 0 : \\v_sld_a0 + 3], v[\\v_sld_offset_a], offset: 0
    ds_read_b128 v[\\v_sld_b0 + 0 : \\v_sld_b0 + 3], v[\\v_sld_offset_b], offset: 0 
    ds_read_b128 v[\\v_sld_b1 + 0 : \\v_sld_b1 + 3], v[\\v_sld_offset_b], offset: 128 * 8 * 2 * 2 * 1
    ds_read_b128 v[\\v_sld_a1 + 0 : \\v_sld_a1 + 3], v[\\v_sld_offset_a], offset: (32 + 0) * 8 * 2 * 2 * 1
    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a0 + 0 : \\v_sld_a0 + 1], v[\\v_sld_b0 + 0 : \\v_sld_b0 + 1], v[\\v_c + 0 : \\v_c + 15]
    ; s_setprio 1
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a0 + 2 : \\v_sld_a0 + 3], v[\\v_sld_b0 + 2 : \\v_sld_b0 + 3], v[\\v_c + 0 : \\v_c + 15]
    ds_read_b128 v[\\v_sld_a0 + 0 : \\v_sld_a0 + 3], v[\\v_sld_offset_a], offset: (32 + 0) * 8 * 2 * 2 * 2
    ds_read_b128 v[\\v_sld_b0 + 0 : \\v_sld_b0 + 3], v[\\v_sld_offset_b], offset: 128 * 8 * 2 * 2 * 2
    s_waitcnt lgkmcnt(2)
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a1 + 0 : \\v_sld_a1 + 1], v[\\v_sld_b1 + 0 : \\v_sld_b1 + 1], v[\\v_c + 0 : \\v_c + 15]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a1 + 2 : \\v_sld_a1 + 3], v[\\v_sld_b1 + 2 : \\v_sld_b1 + 3], v[\\v_c + 0 : \\v_c + 15]
    ; s_setprio 0
    
    ds_read_b128 v[\\v_sld_b1 + 0 : \\v_sld_b1 + 3], v[\\v_sld_offset_b], offset: 128 * 8 * 2 * 2 * 3
    ds_read_b128 v[\\v_sld_a1 + 0 : \\v_sld_a1 + 3], v[\\v_sld_offset_a], offset: (32 + 0) * 8 * 2 * 2 * 3
    s_waitcnt lgkmcnt(2)

    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a0 + 0 : \\v_sld_a0 + 1], v[\\v_sld_b0 + 0 : \\v_sld_b0 + 1], v[\\v_c + 0 : \\v_c + 15]
    ;s_setprio 1
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a0 + 2 : \\v_sld_a0 + 3], v[\\v_sld_b0 + 2 : \\v_sld_b0 + 3], v[\\v_c + 0 : \\v_c + 15]
    s_waitcnt lgkmcnt(0)
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a1 + 0 : \\v_sld_a1 + 1], v[\\v_sld_b1 + 0 : \\v_sld_b1 + 1], v[\\v_c + 0 : \\v_c + 15]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a1 + 2 : \\v_sld_a1 + 3], v[\\v_sld_b1 + 2 : \\v_sld_b1 + 3], v[\\v_c + 0 : \\v_c + 15]
    ;s_setprio 0
.endm
"""
        return MFMA

    def write_wg2x2_macro(self):
        MFMA = """
.macro .mfma_wg2x2_w2x2_32x32x8bf16_1k_ak1_8_bk1_8 v_sld_a0, v_sld_a1, v_sld_b0, v_sld_b1, v_sld_offset_a, v_sld_offset_b, v_c
;    .rept 8
;        v_fmac_f32 v0, v1, v2
;    .endr
;.endm
;.macro fake1
    ds_read_b128 v[\\v_sld_a0 + 0 : \\v_sld_a0 + 3], v[\\v_sld_offset_a], offset: 0
    ds_read_b128 v[\\v_sld_b0 + 0 : \\v_sld_b0 + 3], v[\\v_sld_offset_b], offset: 0 
    ds_read_b128 v[\\v_sld_b1 + 0 : \\v_sld_b1 + 3], v[\\v_sld_offset_b], offset: 64 * 8 * 2 
    ds_read_b128 v[\\v_sld_a1 + 0 : \\v_sld_a1 + 3], v[\\v_sld_offset_a], offset: 64 * 8 * 2

    s_waitcnt lgkmcnt(0)

    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a0 + 0 : \\v_sld_a0 + 1], v[\\v_sld_b0 + 0 : \\v_sld_b0 + 1], v[\\v_c + 0 : \\v_c + 15]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a0 + 2 : \\v_sld_a0 + 3], v[\\v_sld_b0 + 2 : \\v_sld_b0 + 3], v[\\v_c + 0 : \\v_c + 15]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 16 : \\v_c + 31], v[\\v_sld_a0 + 0 : \\v_sld_a0 + 1], v[\\v_sld_b1 + 0 : \\v_sld_b1 + 1], v[\\v_c + 16 : \\v_c + 31]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 16 : \\v_c + 31], v[\\v_sld_a0 + 2 : \\v_sld_a0 + 3], v[\\v_sld_b1 + 2 : \\v_sld_b1 + 3], v[\\v_c + 16 : \\v_c + 31]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 32 : \\v_c + 47], v[\\v_sld_a1 + 0 : \\v_sld_a1 + 1], v[\\v_sld_b0 + 0 : \\v_sld_b0 + 1], v[\\v_c + 32 : \\v_c + 47]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 32 : \\v_c + 47], v[\\v_sld_a1 + 2 : \\v_sld_a1 + 3], v[\\v_sld_b0 + 2 : \\v_sld_b0 + 3], v[\\v_c + 32 : \\v_c + 47]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 48 : \\v_c + 63], v[\\v_sld_a1 + 0 : \\v_sld_a1 + 1], v[\\v_sld_b1 + 0 : \\v_sld_b1 + 1], v[\\v_c + 48 : \\v_c + 63]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 48 : \\v_c + 63], v[\\v_sld_a1 + 2 : \\v_sld_a1 + 3], v[\\v_sld_b1 + 2 : \\v_sld_b1 + 3], v[\\v_c + 48 : \\v_c + 63]

    ds_read_b128 v[\\v_sld_a0 + 0 : \\v_sld_a0 + 3], v[\\v_sld_offset_a], offset: (128 + 0) * 8 * 2 * 2 * 1
    ds_read_b128 v[\\v_sld_b0 + 0 : \\v_sld_b0 + 3], v[\\v_sld_offset_b], offset: 128 * 8 * 2 * 2 * 1
    ds_read_b128 v[\\v_sld_b1 + 0 : \\v_sld_b1 + 3], v[\\v_sld_offset_b], offset: 128 * 8 * 2 * 2 * 1 + 64 * 8 * 2 
    ds_read_b128 v[\\v_sld_a1 + 0 : \\v_sld_a1 + 3], v[\\v_sld_offset_a], offset: (128 + 0) * 8 * 2 * 2 * 1 + 64 * 8 * 2

    s_waitcnt lgkmcnt(0)

    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a0 + 0 : \\v_sld_a0 + 1], v[\\v_sld_b0 + 0 : \\v_sld_b0 + 1], v[\\v_c + 0 : \\v_c + 15]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 0 : \\v_c + 15], v[\\v_sld_a0 + 2 : \\v_sld_a0 + 3], v[\\v_sld_b0 + 2 : \\v_sld_b0 + 3], v[\\v_c + 0 : \\v_c + 15]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 16 : \\v_c + 31], v[\\v_sld_a0 + 0 : \\v_sld_a0 + 1], v[\\v_sld_b1 + 0 : \\v_sld_b1 + 1], v[\\v_c + 16 : \\v_c + 31]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 16 : \\v_c + 31], v[\\v_sld_a0 + 2 : \\v_sld_a0 + 3], v[\\v_sld_b1 + 2 : \\v_sld_b1 + 3], v[\\v_c + 16 : \\v_c + 31]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 32 : \\v_c + 47], v[\\v_sld_a1 + 0 : \\v_sld_a1 + 1], v[\\v_sld_b0 + 0 : \\v_sld_b0 + 1], v[\\v_c + 32 : \\v_c + 47]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 32 : \\v_c + 47], v[\\v_sld_a1 + 2 : \\v_sld_a1 + 3], v[\\v_sld_b0 + 2 : \\v_sld_b0 + 3], v[\\v_c + 32 : \\v_c + 47]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 48 : \\v_c + 63], v[\\v_sld_a1 + 0 : \\v_sld_a1 + 1], v[\\v_sld_b1 + 0 : \\v_sld_b1 + 1], v[\\v_c + 48 : \\v_c + 63]
    v_mfma_f32_32x32x8bf16_1k v[\\v_c + 48 : \\v_c + 63], v[\\v_sld_a1 + 2 : \\v_sld_a1 + 3], v[\\v_sld_b1 + 2 : \\v_sld_b1 + 3], v[\\v_c + 48 : \\v_c + 63]

    s_waitcnt lgkmcnt(0)
    s_barrier
.endm
"""

        return MFMA



