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
        PRINT = """"
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
    def __init__():
