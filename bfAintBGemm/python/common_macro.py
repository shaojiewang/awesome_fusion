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

    ;v_pk_add_f32 v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    ;v_pk_add_f32 v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]

    ;v_pk_mul_f32 v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_tmp + 0 : v_tmp + 1]
    ;v_pk_mul_f32 v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_tmp + 2 : v_tmp + 3]

    v_pk_fma_f32 v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_tmp + 0 : \\v_tmp + 1], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    v_pk_fma_f32 v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_tmp + 2 : \\v_tmp + 3], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]

    v_pack_b32_f16 v[\\v_tmp + 0], v[\\v_tmp + 0], v[\\v_tmp + 1], op_sel: [1, 1]
    v_pack_b32_f16 v[\\v_tmp + 1], v[\\v_tmp + 2], v[\\v_tmp + 3], op_sel: [1, 1]

    v_perm_b32 v[\\v_tmp + 4], v[\\v_base], v[\\v_gld_b + 1], v[\\v_sel_b + 0]
    v_perm_b32 v[\\v_tmp + 5], v[\\v_base], v[\\v_gld_b + 1], v[\\v_sel_b + 1]
    v_perm_b32 v[\\v_tmp + 6], v[\\v_base], v[\\v_gld_b + 1], v[\\v_sel_b + 2]
    v_perm_b32 v[\\v_tmp + 7], v[\\v_base], v[\\v_gld_b + 1], v[\\v_sel_b + 3]

    ;v_pk_add_f32 v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    ;v_pk_add_f32 v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]

    ;v_pk_mul_f32 v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_tmp + 4 : v_tmp + 5]
    ;v_pk_mul_f32 v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_tmp + 6 : v_tmp + 7]

    v_pk_fma_f32 v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_tmp + 4 : \\v_tmp + 5], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    v_pk_fma_f32 v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_tmp + 6 : \\v_tmp + 7], v[\\v_scale + 0 : \\v_scale + 1], v[\\v_sub_magic_num + 0 : \\v_sub_magic_num + 1]
    
    v_pack_b32_f16 v[\\v_tmp + 2], v[\\v_tmp + 4], v[\\v_tmp + 5], op_sel: [1, 1]
    v_pack_b32_f16 v[\\v_tmp + 3], v[\\v_tmp + 6], v[\\v_tmp + 7], op_sel: [1, 1]

.endm
"""
        return DEQUANT




