import gemm_kernel_traits
from dataclasses import dataclass
import common_macro
import kernel_args

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
        print(kernel_str)

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
        print(kernel_str)
        print(k_args.karg_begin_byte)
        
