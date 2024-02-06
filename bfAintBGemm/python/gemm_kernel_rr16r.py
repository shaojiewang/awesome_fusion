import gemm_kernel_traits
from dataclasses import dataclass
import common_macro
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

    def write_kernel(self):
        # macros
        m_print = common_macro.PrintMacro("print")
        print(m_print.macro_body)
 
