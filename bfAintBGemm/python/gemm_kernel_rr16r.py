import gemm_kernel_traits
import common_macro
import kernel_args
import sgprs 
import vgprs 
import amdgpu_metadata
import rodata

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

    def get_lds_size(self) -> int:
        return 65536

    def get_warp_size(self) -> int:
        return 64

    def write_kernel(self):
        # traits
        lds_size = self.get_lds_size()
        warp_size = self.get_warp_size()

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
            "k_ptr_c" : kernel_args.KernelArgTraits(8, 'global_buffer', 'f16', 'global', False),
            "k_ptr_a" : kernel_args.KernelArgTraits(8, 'global_buffer', 'f16', 'global', True),
            "k_ptr_b" : kernel_args.KernelArgTraits(8, 'global_buffer', 'f16', 'global', True),
            "k_ptr_scale" : kernel_args.KernelArgTraits(8, 'global_buffer', 'f32', 'global', True),
            "k_m" : kernel_args.KernelArgTraits(4, 'by_value', 'i32', '', True),
            "k_n" : kernel_args.KernelArgTraits(4, 'by_value', 'i32', '', True),
            "k_k" : kernel_args.KernelArgTraits(4, 'by_value', 'i32', '', True),
            "k_lda" : kernel_args.KernelArgTraits(4, 'by_value', 'i32', '', True),
            "k_ldb" : kernel_args.KernelArgTraits(4, 'by_value', 'i32', '', True),
            "k_ldc" : kernel_args.KernelArgTraits(4, 'by_value', 'i32', '', True),
            "k_k_per_cta" : kernel_args.KernelArgTraits(4, 'by_value', 'i32', '', True),
            "k_print" : kernel_args.KernelArgTraits(8, 'global_buffer', 'f32', 'global', False),
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
        # print(kernel_str)

        # vgprs
        dict_vgprs = {
            "v_c" : 16,
            "v_sld_a0" : 4,
            "v_sld_b0" : 4,
            "v_sld_a1" : 4,
            "v_sld_b1" : 4,
            "v_gld_a0" : 4,
            "v_gld_a1" : 4,
            "v_gld_b0" : 8,
            "v_gld_b1" : 8,
            "v_lane_id" : 1,
            "v_offset_a_k0" : 1,
            "v_offset_a" : 1,
            "v_offset_b_k0" : 1,
            "v_offset_b" : 1,
            "v_lane_im" : 1,
            "v_lane_in" : 1,
            "v_sst_offset_c" : 1,
            "v_iak0" : 1,
            "v_im" : 1,
            "v_ibk0" : 1,
            "v_in" : 1,
            "v_sst_offset_a0" : 1,
            "v_sst_offset_a1" : 1,
            "v_sst_offset_b0" : 1,
            "v_sst_offset_b1" : 1,
            "v_sld_iak0" : 1,
            "v_sld_im" : 1,
            "v_sld_offset_a0" : 1,
            "v_sld_offset_a1" : 1,
            "v_sld_ibk0" : 1,
            "v_sld_in" : 1,
            "v_sld_offset_b0" : 1,
            "v_sld_offset_b1" : 1,
            "v_c_in" : 1,
            "v_c_im" : 1,
            "v_sld_offset_c" : 1,
            "v_gst_offset_c" : 1,
            "v_fp32_base" : 1,
            "v_sel_b" : 4,
            "v_sub_magic_num" : 2,
            "v_scale" : 2,
            "v_c_n_flag" : 1,
            "v_c_cur_m" : 1,
            "v_tid" : 1,
            "v_wave_id" : 1,
            "v_tmp" : 8,
        }
        k_vgprs = vgprs.Vgprs(**dict_vgprs)
        kernel_str += k_vgprs.vgprs_body
        #print(kernel_str)

        # rodata
        rod = rodata.Rodata(
            "bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipelined_splitk",
            lds_size,
            0,
            1,
            1,
            1,
            1,
            0,
            k_vgprs.vgpr_offset,
            k_sgprs.sgpr_offset,
            0,
            0,
            k_vgprs.vgpr_offset)
        k_rodata = rod.rodata_str
        print(k_rodata)

        # metadata
        md = amdgpu_metadata.AmdgpuMetadata(
            [1, 0],
            "bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipelined_splitk",
            k_sgprs.sgpr_offset,
            k_vgprs.vgpr_offset,
            8,
            k_args.kargs_offset,
            lds_size,
            0,
            warp_size,
            [256, 1, 1],
            256,
            dict_kernel_args)
        k_amdgpu_metadata = md.metadata_body
        #print(md.metadata_body)


