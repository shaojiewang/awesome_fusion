import math
import os
import subprocess

import gemm_kernel_traits
import common_macro
import kernel_args
import sgprs 
import vgprs 
import amdgpu_metadata
import rodata
import text_seg

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

    def get_kernel_name(self) -> str:
        return "bf16gemm_rr16r_b256_32x128x64_wg1x1_w1x4_32x32x8bf16_1k_pregld1_pipelined_splitk"

    def get_asm_file_name(self) -> str:
        return self.get_kernel_name() + ".s"

    def get_hsaco_name(self) -> str:
        return self.get_kernel_name() + ".hsaco"

    def gen_kernel_label(self) -> str:
        return self.get_kernel_name() + ": \n" + \
            "    ; http://www.hsafoundation.com/html/Content/Runtime/Topics/02_Core/hsa_kernel_dispatch_packet_t.htm\n"

    def gen_program_end(self) -> str:
        end_p_str = "l_end_" + self.get_kernel_name() + ": \n"
        end_p_str += "    ; .print v_offset_a, s_print, s_bx, v_tid, v_tmp + 7\n"
        end_p_str += "    s_endpgm"
        return end_p_str

    def gen_kargs_load(self) -> str:
        kargs_load_str = """
    s_load_dwordx2 s[s_ptr_c:s_ptr_c+1], s[s_ka:s_ka+1], 0+k_ptr_c
    s_load_dwordx2 s[s_ptr_a:s_ptr_a+1], s[s_ka:s_ka+1], 0+k_ptr_a
    s_load_dwordx2 s[s_ptr_b:s_ptr_b+1], s[s_ka:s_ka+1], 0+k_ptr_b
    s_load_dwordx2 s[s_ptr_scale:s_ptr_scale+1], s[s_ka:s_ka+1], 0+k_ptr_scale
    s_load_dwordx2 s[s_print:s_print+1], s[s_ka:s_ka+1], 0+k_print

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
"""
        return kargs_load_str

    def gen_ld_a_b_c(self):
        ldabc_str = "\n    ; adjust lda/b/c according to the datatypes\n"
        if self.a_datatype.data_size != 1:
            log2_data_size = int(math.log2(self.a_datatype.data_size))
            ldabc_str += "    s_lshl_b32 s[s_lda], s[s_lda], {}\n".format(log2_data_size)
        
        if self.b_datatype.data_size != 1:
            log2_data_size = int(math.log2(self.b_datatype.data_size))
            ldabc_str += "    s_lshl_b32 s[s_ldb], s[s_ldb], {}\n".format(log2_data_size)

        if self.c_datatype.data_size != 1:
            log2_data_size = int(math.log2(self.c_datatype.data_size))
            ldabc_str += "    s_lshl_b32 s[s_ldc], s[s_ldc], {}\n".format(log2_data_size)
        return ldabc_str

    def gen_cta_mapping(self):
        CTA_MAP = """
    ; thread block mapping
    ; m block id: bid x
    ; n block id: bid y
    ; k block id: bid z
    s_mul_i32 s[s_m_idx], s[s_bx], {}
    s_mul_i32 s[s_n_idx], s[s_by], {}
    s_mul_i32 s[s_k_idx], s[s_bz], s[s_k_per_cta]

"""
        cta_map_str = CTA_MAP.format(self.tile.cta_m, self.tile.cta_n)
        return cta_map_str

    def gen_kernel(self):
        # traits
        lds_size = self.get_lds_size()
        warp_size = self.get_warp_size()
        kernel_name = self.get_kernel_name()

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
            kernel_name,
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
        #print(k_rodata)

        # metadata
        md = amdgpu_metadata.AmdgpuMetadata(
            [1, 0],
            kernel_name, 
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

        # text segment
        txt_str = text_seg.TextSeg(kernel_name, 8)
        kernel_str += txt_str.text_seg_str

        # kernel label
        k_label = self.gen_kernel_label()
        kernel_str += k_label

        # kernel args load
        kargs_load_inst = self.gen_kargs_load()
        kernel_str += kargs_load_inst 

        # ld a/b/c
        ldabc_str = self.gen_ld_a_b_c()
        kernel_str += ldabc_str

        # cta mapping
        cta_map_str = self.gen_cta_mapping()
        kernel_str += cta_map_str

        # program end
        p_end_str = self.gen_program_end() 
        kernel_str += p_end_str
        kernel_str += k_rodata
        kernel_str += k_amdgpu_metadata


        print(kernel_str)
        return kernel_str

    def write_kernel(self, output_dir):
        if os.path.exists(output_dir):
            pass
        else:
            os.mkdirs(output_dir)

        kernel_str = self.gen_kernel()
        asm_name = self.get_asm_file_name()
        asm_file_name = os.path.join(output_dir, asm_name)
        with open(asm_file_name, "w") as asm_f:
            asm_f.write(kernel_str)

    def compile_kernel(self, output_dir):
        asm_name = self.get_asm_file_name()
        asm_path = os.path.join(output_dir, asm_name)

        hsaco_name = self.get_hsaco_name()
        hsaco_path = os.path.join(output_dir, hsaco_name)

        if os.path.exists(asm_path):
            compile_cmd = ['/opt/rocm/llvm/bin/clang++']
            compile_cmd.append('-x')
            compile_cmd.append('assembler')
            compile_cmd.append('-target')
            compile_cmd.append('amdgcn--amdhsa')
            compile_cmd.append('-mcpu=gfx90a')
            compile_cmd.append(asm_path)
            compile_cmd.append('-o')
            compile_cmd.append(hsaco_path)
            subprocess.run(compile_cmd, stdout=subprocess.PIPE)
        else:
            assert false, "{} file is not generated yet".format(asm_path)

    

