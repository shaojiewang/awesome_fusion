import gemm_kernel_traits
from dataclasses import dataclass
import common_macro
import kernel_args
import sgprs 

@dataclass
class GemmKernelRR16R(gemm_kernel_traits.GemmKernelTraits):
    def __init__(self, 
                 a_datatype, 
                 b_datatype, 
                 c_datatype, 
                 scale_datatype, 
                 splitk, 
                 gemm_tile : gemm_kernel_traits.GemmTileSize, 
                 pipeline="v1"):
        super(GemmKernelRR16R, self).__init__("row",
                                              "row",
                                              "row",
                                              1,
                                              16,
                                              1,
                                              a_datatype,
                                              b_datatype,
                                              c_datatype,
                                              scale_datatype,
                                              splitk,
                                              gemm_tile,
                                              pipeline)
        self.kernel_body = ""

    def write_kernel(self):
        # macros
        kernel_str = ""
        m_print = common_macro.PrintMacro("print")
        m_dequant = common_macro.DequantMacro("dequant")
        m_mfma = common_macro.MfmaMacro("mfma")
        kernel_str += m_print.macro_body
        kernel_str += m_dequant.macro_body
        kernel_str += m_mfma.macro_body
        # print(kernel_str)

        # kernel args
        dict_kernel_args = {
            "k_ptr_c" : 2,
            "k_ptr_a" : 2,
            "k_ptr_b" : 2,
            "k_ptr_scale" : 2,
            "k_m" : 1,
            "k_n" : 1,
            "k_k" : 1,
            "k_lda" : 1,
            "k_ldb" : 1,
            "k_ldc" : 1,
            "k_k_per_cta" : 1,
            "k_print" : 2
        }
        k_args = kernel_args.KernelArgs(**dict_kernel_args)
        kernel_str += k_args.kargs_body
        # print(kernel_str)
        # print(k_args.karg_begin_byte)

        # sgprs
        dict_sgprs = {
            "s_ka" : 2,
            "s_bx" : 1,
            "s_by" : 1,
            "s_bz" : 1,
            "s_ptr_c" : 4,
            "s_ptr_a" : 4,
            "s_ptr_b" : 4,
            "s_ptr_scale" : 4,
            "s_m" : 1,
            "s_n" : 1,
            "s_k" : 1,
            "s_lda" : 1,
            "s_ldb" : 1,
            "s_ldc" : 1,
            "s_k_per_cta" : 1,
            "s_print" : 2,
            "s_bs_a" : 1,
            "s_bs_b" : 1,
            "s_m_blocks" : 1,
            "s_m_idx" : 1,
            "s_n_idx" : 1,
            "s_offset_a" : 1,
            "s_offset_b" : 4,
            "s_kitr" : 1,
            "s_wave_id" : 1,
            "s_wave_im" : 1,
            "s_wave_in" : 1,
            "s_k_idx" : 1,
            "s_tmp" : 8,
        }
        k_sgprs = sgprs.Sgprs(**dict_sgprs)
        kernel_str += k_sgprs.sgprs_body
        print(kernel_str)
