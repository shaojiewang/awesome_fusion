import math
import os
import subprocess

import gemm_kernel_traits
import common_funcs
import common_macro
import kernel_args
import sgprs 
import vgprs 
import amdgpu_metadata
import rodata
import text_seg
import datatype
import pipeline_selector
import pipeline_1x1_interleaved
import pipeline_2x2_interleaved
import c_write_out

class GemmKernelRR16R(gemm_kernel_traits.GemmKernelTraits):
    def __init__(self, 
                 a_datatype, 
                 b_datatype, 
                 c_datatype, 
                 scale_datatype, 
                 acc_datatype,
                 compute_datatype,
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

        cta_size = gemm_tile.cta_size
        
        self.b_n = gemm_tile.cta_n
        self.b_bk0 = cta_size // self.b_n
        self.b_bk1 = 1
        self.t_bk1 = gemm_tile.global_bk1
        self.t_bk0 = gemm_tile.cta_k // (self.t_bk1 * self.b_bk0 * self.b_bk1)
        self.t_n = 1

        self.t_ak1 = gemm_tile.gmem_vec_a
        self.b_ak1 = 1
        self.t_ak0 = 1
        self.b_ak0 = gemm_tile.cta_k // (self.t_ak1 * self.b_ak1 * self.t_ak0)
        self.b_m = gemm_tile.cta_size // self.b_ak0
        self.t_m = gemm_tile.cta_m // self.b_m

        self.thread_vec_scale = [1, 1, 1]
        self.block_vec_scale  = [self.b_bk0, self.b_n, self.b_bk1]

        self.thread_vec_a = [self.t_ak0, self.t_m, self.t_ak1]
        self.block_vec_a  = [self.b_ak0, self.b_m, self.b_ak1]
        self.thread_vec_b = [self.t_bk0, self.t_n, self.t_bk1]
        self.block_vec_b  = [self.b_bk0, self.b_n, self.b_ak1]

        self.num_warp_n = self.tile.warp_n // self.tile.inst_n
        self.num_warp_m = self.tile.warp_m // self.tile.inst_m

        self.acc_gpr_group = 4
        self.acc_num = self.tile.cta_n * self.tile.cta_m // self.tile.cta_size
        self.acc_datatype = acc_datatype
        self.compute_datatype = compute_datatype

        self.b_cn = self.tile.cta_n // self.tile.gmem_vec_c

        self.smem_a_padding = 1

        # wg repeat and number m/n in wave
        self.num_wave_m = gemm_tile.warp_m // gemm_tile.inst_m
        self.num_wave_n = gemm_tile.warp_n // gemm_tile.inst_n
        self.wg_repeat_m = gemm_tile.cta_m // gemm_tile.warp_m
        self.wg_repeat_n = gemm_tile.cta_n // gemm_tile.warp_n
        
        # pipeline selector
        if pipeline == "v1":
            if self.wg_repeat_m == 1 and self.wg_repeat_n == 1:
                self.pipeline = pipeline_selector.k_pipeline_1x1_lds_double_buffer_interleaved
            if self.wg_repeat_m == 2 and self.wg_repeat_n == 2:
                self.pipeline = pipeline_selector.k_pipeline_2x2_interleaved

    def get_a_smem_size(self) -> int:
        cta_m = self.tile.cta_m
        cta_k = self.tile.cta_k
        smem_a_padding = self.smem_a_padding
        a_smem_size = (cta_m + smem_a_padding) * cta_k * self.a_datatype.data_size
        return int(a_smem_size)

    def get_lds_size(self) -> int:
        return 65536

    def get_warp_size(self) -> int:
        return 64

    def get_kernel_name(self) -> str:
        KERNEL_NAME = """bf16gemm_rr{F_bk1}r_b{F_cta_size}_{F_cta_m}x{F_cta_n}x{F_cta_k}_wg{F_repeat_m}x{F_repeat_n}_w{F_wave_num_m}x{F_wave_num_n}_{F_inst_m}x{F_inst_n}x{F_inst_k}bf16_1k_pregld1_pipeline_interleaved_splitk"""
        name_str = KERNEL_NAME.format(F_bk1=self.tile.global_bk1,
                                      F_cta_size=self.tile.cta_size,
                                      F_cta_m=self.tile.cta_m,
                                      F_cta_n=self.tile.cta_n,
                                      F_cta_k=self.tile.cta_k,
                                      F_repeat_m=self.wg_repeat_m,
                                      F_repeat_n=self.wg_repeat_n,
                                      F_wave_num_m=self.num_wave_m,
                                      F_wave_num_n=self.num_wave_n,
                                      F_inst_m=self.tile.inst_m,
                                      F_inst_n=self.tile.inst_n,
                                      F_inst_k=self.tile.inst_k)
        return name_str

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

    def get_sgpr_dict(self) -> dict:
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
            "s_offset_a" : 4,
            "s_offset_b" : 1,
            "s_kitr" : 1,
            "s_wave_id" : 1,
            "s_wave_im" : 1,
            "s_wave_in" : 1,
            "s_k_idx" : 1,
            "s_tmp" : 8,
        }
        return dict_sgprs
        
    def get_vgpr_dict(self) -> dict:
        acc_num = self.tile.cta_m * self.tile.cta_n // self.tile.cta_size
        ele_per_compute_vgpr = 4 // (self.compute_datatype.data_size)
        ele_per_a_vgpr = 4 // (self.a_datatype.data_size)
        ele_per_b_vgpr = 4 // (self.b_datatype.data_size)
        sld_a_num = self.tile.inst_m * 2 * self.tile.inst_k // (self.get_warp_size() * ele_per_compute_vgpr)
        sld_b_num = self.tile.inst_n * 2 * self.tile.inst_k // (self.get_warp_size() * ele_per_compute_vgpr)
        gld_a_num = self.tile.cta_m * self.tile.cta_k // (self.tile.cta_size * ele_per_a_vgpr)
        gld_b_num = self.tile.cta_n * self.tile.cta_k // (self.tile.cta_size * ele_per_b_vgpr)
        dict_vgprs = {
            "v_c" : acc_num,
            "v_sld_a0" : sld_a_num,
            "v_sld_b0" : sld_b_num,
            "v_sld_a1" : sld_a_num,
            "v_sld_b1" : sld_b_num,
            "v_gld_a0" : gld_a_num,
            "v_gld_a1" : gld_a_num,
            "v_gld_b0" : gld_b_num,
            "v_gld_b1" : gld_b_num,
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
        return dict_vgprs

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

    def gen_scale_load(self):
        ADDRCALC = """
    ; load scale
    ; TODO: to avoid cache line waste
    ; Scale:
    ; thread vec: [n]         = [{F_t_n}]
    ; block vec:  [k0, n, k1] = [{F_b_k0},{F_b_n},{F_b_k1}]
    v_mov_b32 v[v_tmp], {F_b_n_minus_1}
    v_and_b32 v[v_tmp], v[v_tid], v[v_tmp]
    v_lshlrev_b32 v[v_tmp], {F_sizeof_type}, v[v_tmp]
    s_lshl_b32 s[s_tmp], s[s_n_idx], {F_sizeof_type}
    s_add_u32  s[s_ptr_scale], s[s_ptr_scale], s[s_tmp]
    s_addc_u32 s[s_ptr_scale + 1], s[s_ptr_scale + 1], 0
    s_lshl_b32 s[s_ptr_scale + 2], s[s_n], {F_sizeof_type}
    s_sub_i32 s[s_ptr_scale + 2], s[s_ptr_scale + 2], s[s_tmp]
"""
        GLDDWORD = """
    buffer_load_dword v[v_scale], v[v_tmp], s[s_ptr_scale : s_ptr_scale + 3], 0 offen offset: {F_offset}
"""
        t_n = self.thread_vec_scale[1]
        b_k0 = self.block_vec_scale[0]
        b_n = self.block_vec_scale[1]
        log_sizeof_scale = int(math.log2(self.scale_datatype.data_size))
        k_src = ""
        k_src += ADDRCALC.format(F_t_n=t_n, F_b_k0=b_k0, F_b_k1=1, F_b_n=b_n, F_b_n_minus_1=b_n - 1, F_sizeof_type=log_sizeof_scale)
        if t_n == 1:
            k_src += GLDDWORD.format(F_offset=0)
        return k_src

    def gen_a_matrix_gld_addr(self):
        ADDRCALC = """
    ; load A matrix
    ; A:
    ; thread vec: [ak0, m, ak1] = [{F_t_ak0}, {F_t_m}, {F_t_ak1}]
    ; block vec:  [ak0, m, ak1] = [{F_b_ak0}, {F_b_m}, {F_b_ak1}]

    ; A thread block offset
    v_and_b32 v[v_iak0], v[v_tid], {F_b_ak0_minus_1}
    v_lshrrev_b32 v[v_im], {F_log2_b_ak0}, v[v_tid]
    v_lshlrev_b32 v[v_tmp], {F_log2_t_ak1}, v[v_iak0]
    v_mad_u32_u24 v[v_offset_a], v[v_im], s[s_lda], v[v_tmp]
    ; A grid offset
    s_mul_i32 s[s_tmp], s[s_m_idx], s[s_lda]
    s_lshl_b32 s[s_tmp + 1], s[s_k_idx], {F_log2_sizeof_type}
    s_add_i32 s[s_tmp], s[s_tmp], s[s_tmp + 1]
    s_add_u32  s[s_ptr_a], s[s_ptr_a], s[s_tmp]
    s_addc_u32 s[s_ptr_a + 1], s[s_ptr_a + 1], 0
    ; prefetch load A
    s_mul_i32 s[s_ptr_a + 2], s[s_m], s[s_lda]
    s_sub_i32 s[s_ptr_a + 2], s[s_ptr_a + 2], s[s_tmp]

    s_mov_b32 s[s_bs_a], {F_move_step}
"""
        t_ak0, t_m, t_ak1 = self.thread_vec_a[0], self.thread_vec_a[1], self.thread_vec_a[2]
        b_ak0, b_m, b_ak1 = self.block_vec_a[0], self.block_vec_a[1], self.block_vec_a[2]
        log2_b_ak0 = int(math.log2(b_ak0))
        log2_t_ak1 = int(math.log2(t_ak1 * self.a_datatype.data_size))
        log2_sizeof_dt = int(math.log2(self.a_datatype.data_size))
        move_step = self.tile.cta_k * self.a_datatype.data_size
        a_addr_calc = ADDRCALC.format(F_t_ak0=t_ak0, F_t_m=t_m, F_t_ak1=t_ak1, 
                                      F_b_ak0=b_ak0, F_b_m=b_m, F_b_ak1=b_ak1, 
                                      F_b_ak0_minus_1=b_ak0 - 1, F_log2_b_ak0=log2_b_ak0, 
                                      F_log2_t_ak1=log2_t_ak1, F_log2_sizeof_type=log2_sizeof_dt,
                                      F_move_step=move_step)

        OFFSET = """
    s_mul_i32 s[s_offset_a + {F_soffset_idx}], s[s_lda], {F_idx}
"""
        for i in range(1, t_m, 1):
            a_addr_calc += OFFSET.format(F_soffset_idx=i - 1, F_idx=i * b_m)
            
        return a_addr_calc

    def gen_a_matrix_gld_inst(self, v_gld_a):
        GLDDWORDX4 = """
    buffer_load_dwordx4 v[{F_v_gld_a} + {F_vgpr_b} : {F_v_gld_a} + {F_vgpr_e}], v[v_offset_a], s[s_ptr_a : s_ptr_a + 3], {F_s_offset} offen offset:0"""
        MOVE_STEP = """
    v_add_u32 v[v_offset_a], v[v_offset_a], s[s_bs_a]
"""
        a_gld_src = ""
        t_m = self.thread_vec_a[1]
        for i in range(t_m):
            soff_idx = i - 1
            s_offset = 0 if i == 0 else "s[s_offset_a + {F_soff_idx}]".format(F_soff_idx=soff_idx)
            a_gld_src += GLDDWORDX4.format(F_v_gld_a=v_gld_a, F_vgpr_b=i * 4, F_vgpr_e=i * 4 + 3, F_s_offset=s_offset)
        a_gld_src += MOVE_STEP

        return a_gld_src

    def gen_b_matrix_gld_addr(self):
        ADDRCALC = """
    ; load B matrix
    ; B:
    ; thread vec: [bk0, n, bk1] = [{F_t_bk0}, {F_t_n}, {F_t_bk1}]
    ; block vec:  [bk0, n, bk1] = [{F_b_bk0}, {F_b_n}, {F_b_bk1}]
    ; B thread block offset
    v_mov_b32 v[v_tmp], {F_b_n_minus_1}
    v_and_b32 v[v_in], v[v_tid], v[v_tmp]
    v_lshrrev_b32 v[v_ibk0], {F_log2_b_n}, v[v_tid]
    v_lshlrev_b32 v[v_tmp], {F_log2_t_bk1}, v[v_in]
    ; k0 offset = ldb
    v_mad_u32_u24 v[v_offset_b], v[v_ibk0], s[s_ldb], v[v_tmp]
    ; B grid offset
    s_lshr_b32 s[s_tmp + 1], s[s_ldb], {F_log2_t_bk1}
    s_lshl_b32 s[s_tmp], s[s_n_idx], {F_log2_t_bk1}
    s_mul_i32 s[s_tmp + 2], s[s_tmp + 1], s[s_k_idx]
    s_add_u32 s[s_tmp], s[s_tmp], s[s_tmp + 2]
    s_add_u32  s[s_ptr_b], s[s_ptr_b], s[s_tmp]
    s_addc_u32 s[s_ptr_b + 1], s[s_ptr_b + 1], 0
    ; prefetch load B
    s_mul_i32 s[s_ptr_b + 2], s[s_k], s[s_tmp + 1]
    s_sub_i32 s[s_ptr_b + 2], s[s_ptr_b + 2], s[s_tmp]
    s_lshl_b32 s[s_bs_b], s[s_ldb], {F_log2_k0}
"""
        t_bk0, t_n, t_bk1 = self.thread_vec_b[0], self.thread_vec_b[1], self.thread_vec_b[2]
        b_bk0, b_n, b_bk1 = self.block_vec_b[0], self.block_vec_b[1], self.block_vec_b[2]
        log2_b_n = int(math.log2(b_n))
        log2_t_bk1 = int(math.log2(t_bk1 * self.b_datatype.data_size))
        log2_sizeof_dt = int(math.log2(self.b_datatype.data_size))
        log2_k0 = int(math.log2(b_bk0 * t_bk0))
        addr_src = ADDRCALC.format(F_t_bk0=t_bk0, F_t_n=t_n, F_t_bk1=t_bk1,
                                   F_b_bk0=b_bk0, F_b_n=b_n, F_b_bk1=b_bk1,
                                   F_b_n_minus_1=b_n - 1, F_log2_b_n=log2_b_n,
                                   F_log2_t_bk1=log2_t_bk1, F_log2_k0=log2_k0)

        OFFSET = """
    s_mul_i32 s[s_offset_b + {F_soffset_idx}], s[s_ldb], {F_idx}
"""
        for i in range(1, t_bk0, 1):
            addr_src += OFFSET.format(F_soffset_idx=i - 1, F_idx=i * b_bk0)

        return addr_src

    def gen_b_matrix_gld_inst(self, v_gld_b):
        GLDDWORDX4 = """
    buffer_load_dwordx4 v[{F_v_gld_b} + {F_vpgr_b} : {F_v_gld_b} + {F_vgpr_e}], v[v_offset_b], s[s_ptr_b : s_ptr_b + 3], {F_s_offset} offen offset:0"""
        gld_src = ""
        t_bk0 = self.thread_vec_b[0]
        for i in range(t_bk0):
            soff_idx = i - 1
            s_offset = 0 if i == 0 else "s[s_offset_b + {F_soff_idx}]".format(F_soff_idx=soff_idx)
            gld_src += GLDDWORDX4.format(F_v_gld_b=v_gld_b, F_vpgr_b=i * 4, F_vgpr_e=i * 4 + 3, F_s_offset=s_offset)

        MOVESTEP = """
    v_add_u32 v[v_offset_b], v[v_offset_b], s[s_bs_b]
"""
        gld_src += MOVESTEP

        return gld_src
        
    def gen_c_gst_addr(self):
        COMMENT = """
    ; store C offset
    ; vgpr to lds
    ; vgpr_group  = 4
    ; wave_id = tid / wave_size
    ; lane_id = tid % wave_size
    ; lane_in = lane_id % inst_n = tid % inst_n
    ; lane_im = lane_id / inst_n * vgpr_group
    ; wave_n = block_n / inst_n
    ; wave_m = block_m / inst_m
    ; wave_in = wave_id % wave_n
    ; wave_im = wave_id / wave_n
""" 

        WAVEID = """
    ; wave id
    v_lshrrev_b32 v[v_wave_id], 6, v[v_tid]
    v_readfirstlane_b32 s[s_wave_id], v[v_wave_id]
    s_lshr_b32 s[s_wave_im], s[s_wave_id], {F_log2_num_wave_n}
    s_and_b32  s[s_wave_in], s[s_wave_id], {F_num_wave_n_minus_1}
    s_lshl_b32 s[s_wave_im], s[s_wave_im], {F_log2_inst_m}
    s_lshl_b32 s[s_wave_in], s[s_wave_in], {F_log2_inst_n}
"""
        log2_num_wave_n = int(math.log2(self.num_warp_n))
        num_wave_n_minus_1 = self.num_warp_n - 1
        log2_inst_m = int(math.log2(self.tile.inst_m))
        log2_inst_n = int(math.log2(self.tile.inst_n))

        wave_id_src = WAVEID.format(F_log2_num_wave_n=log2_num_wave_n, F_num_wave_n_minus_1=num_wave_n_minus_1, F_log2_inst_m=log2_inst_m, F_log2_inst_n=log2_inst_n)

        LANEID = """
    ; lane id
    v_and_b32 v[v_lane_id], 63, v[v_tid]
    v_and_b32 v[v_lane_in], {F_inst_n_minus_1}, v[v_tid] 
    v_lshrrev_b32 v[v_lane_im], {F_log2_inst_n}, v[v_lane_id]
    v_lshlrev_b32 v[v_lane_im], {F_log2_vgpr_group}, v[v_lane_im]
"""
        inst_n_minus_1 = self.tile.inst_n - 1
        log2_vgpr_group = int(math.log2(self.acc_gpr_group))
        
        lane_id_str = LANEID.format(F_inst_n_minus_1=inst_n_minus_1, F_log2_inst_n=log2_inst_n, F_log2_vgpr_group=log2_vgpr_group)

        SST_C_OFFSET = """
    ; sst offset C
    ; m_offset = (wave_im + lane_im) * block_n
    ; n_offset = wave_n + lane_in
    ; sst_c_offset = m_offset + n_offset
    v_add_lshl_u32 v[v_sst_offset_c], v[v_lane_im], s[s_wave_im], {F_log2_inst_m_size}
    v_add_u32 v[v_tmp], v[v_lane_in], s[s_wave_in]
    v_add_lshl_u32 v[v_sst_offset_c], v[v_tmp], v[v_sst_offset_c], {F_log2_sizeof_dt}
"""

        log2_inst_m_size = int(math.log2(self.tile.inst_m * self.acc_datatype.data_size))
        log2_sizeof_dt = int(math.log2(self.c_datatype.data_size))
        sst_c_offset_src = SST_C_OFFSET.format(F_log2_inst_m_size=log2_inst_m_size, F_log2_sizeof_dt=log2_sizeof_dt)

        SLD_GST_C_OFFSET = """
    ; sld/gst offset C
    ; c_in = tid % (block_n / vec_c_n)
    ; c_im = tid / (block_n / vec_c_n)
    ; sld_c_offset = c_in * vec_c_n + c_im * block_n
    ; gst_c_offset = c_in * vec_c_n + c_im * ldc
    v_and_b32 v[v_c_in], {F_b_cn_minus_1}, v[v_tid]
    v_lshrrev_b32 v[v_c_im], {F_log2_b_cn}, v[v_tid]
    v_lshlrev_b32 v[v_tmp], {F_log2_t_cn_byte}, v[v_c_in]
    v_lshl_add_u32 v[v_sld_offset_c], v[v_c_im], {F_log2_cta_n_byte}, v[v_tmp]
    v_mul_lo_u32 v[v_tmp + 1], v[v_c_im], s[s_ldc]
    v_add_u32 v[v_gst_offset_c], v[v_tmp + 1], v[v_tmp]
"""
        b_cn_minus_1 = self.b_cn - 1
        log2_b_cn = int(math.log2(self.b_cn))
        log2_t_cn_byte = int(math.log2(self.c_datatype.data_size * self.tile.gmem_vec_c))
        log2_cta_n_byte = int(math.log2(self.c_datatype.data_size * self.tile.cta_n))

        sld_gst_c_offset = SLD_GST_C_OFFSET.format(F_b_cn_minus_1=b_cn_minus_1, F_log2_b_cn=log2_b_cn, F_log2_t_cn_byte=log2_t_cn_byte, F_log2_cta_n_byte=log2_cta_n_byte)

        C_GRID_POINTER = """
    ; c grid pointer
    s_mul_i32 s[s_tmp], s[s_m_idx], s[s_ldc]
    s_lshl_b32 s[s_tmp + 2], s[s_n_idx], {F_log2_sizeof_dt}
    s_add_u32 s[s_tmp + 1], s[s_tmp + 2], s[s_tmp]
    s_mul_i32 s[s_tmp], s[s_m], s[s_ldc]
    s_mul_i32 s[s_tmp], s[s_tmp], s[s_bz]
    s_add_u32 s[s_ptr_c], s[s_ptr_c], s[s_tmp + 1]
    s_addc_u32 s[s_ptr_c + 1], s[s_ptr_c + 1], 0
    s_add_u32 s[s_ptr_c], s[s_ptr_c], s[s_tmp]
    s_addc_u32 s[s_ptr_c + 1], s[s_ptr_c + 1], 0
    s_mul_i32 s[s_ptr_c + 2], s[s_m], s[s_ldc]
    s_sub_i32 s[s_ptr_c + 2], s[s_ptr_c + 2], s[s_tmp + 1]
    ; c n flag
    v_lshl_add_u32 v[v_tmp], v[v_c_in], {F_log2_t_cn}, s[s_n_idx]
    v_cmp_gt_u32 vcc, s[s_n], v[v_c_in]
    v_cndmask_b32 v[v_c_n_flag],  0, 1, vcc
"""
        log2_t_cn = int(math.log2(self.tile.gmem_vec_c))

        c_grid_src = C_GRID_POINTER.format(F_log2_sizeof_dt=log2_sizeof_dt, F_log2_t_cn=log2_t_cn)

        inst_src = COMMENT + wave_id_src + lane_id_str + sst_c_offset_src + sld_gst_c_offset + c_grid_src
        return inst_src

    def gen_sst_a_offset(self):
        SST_A_OFFSET = """
    ; store A to shared mem offset
    ; sst_iak0 = iak0 * (block_m + pad) * ak1
    ; sst_offset_a = sst_iak0 + v_im * {F_smem_ak1}
    v_lshlrev_b32 v[v_tmp], {F_log2_smem_ak1_byte}, v[v_im]
    v_mov_b32 v[v_tmp + 1], {F_smem_a_line_byte}
    ;v_lshrrev_b32 v[v_tmp + 2], 1, v[v_iak0]
    ;v_and_b32 v[v_tmp + 3], 1, v[v_iak0]
    ;v_lshlrev_b32 v[v_tmp + 3], 3, v[v_tmp + 3]
    ;v_add_u32 v[v_tmp], v[v_tmp], v[v_tmp + 3]
    v_mad_u32_u24 v[v_sst_offset_a0], v[v_iak0], v[v_tmp + 1], v[v_tmp]
    v_mov_b32 v[v_tmp], 0x8000
    v_xor_b32 v[v_sst_offset_a1], v[v_tmp], v[v_sst_offset_a0]
"""
        smem_ak1 = self.tile.smem_a_k1
        smem_ak1_byte = smem_ak1 * self.a_datatype.data_size
        log2_smem_ak1_byte = int(math.log2(smem_ak1_byte))
        smem_a_line_byte = (self.tile.cta_m + self.smem_a_padding) * smem_ak1 * self.a_datatype.data_size
        sst_a_offset_src = SST_A_OFFSET.format(F_smem_ak1=smem_ak1, F_log2_smem_ak1_byte=log2_smem_ak1_byte, F_smem_a_line_byte=smem_a_line_byte)
        return sst_a_offset_src

    def gen_sst_b_offset(self):
        SST_B_OFFSET = """
    ; store B to shared mem offset. when B is stored to shared mem, B datatype is bf16/fp16
    ; bk1 = max(ak1, bk1_gld, 8)
    ; sst_in = v_in * bk1 * n1 = v_in * 8 * 1
    ; sst_ibk0 = v_ibk0 * block_n * bk1_gld = v_ibk0 * {F_cta_n} * {F_global_bk1}
    ; sst_offset_b = sst_in + sst_ibk0
    ; padding = sst_offset_b / 64 * 8
    ; sst_offset_b = sst_offset_b + padding
    v_lshlrev_b32 v[v_tmp], {F_log2_smem_b_k1}, v[v_in]
    v_lshlrev_b32 v[v_tmp + 1], {F_log2_smem_bk0_stride}, v[v_ibk0]
    v_add_u32 v[v_sst_offset_b0], v[v_tmp], v[v_tmp + 1]
    ; v_lshrrev_b32 v[v_tmp], 6, v[v_sst_offset_b]
    ; v_lshl_add_u32 v[v_sst_offset_b], v[v_tmp], 3, v[v_sst_offset_b] 
    v_lshlrev_b32 v[v_sst_offset_b0], {F_log2_compute_dt_size}, v[v_sst_offset_b0]
    v_mov_b32 v[v_tmp], {F_a_smem_size}
    v_add_u32 v[v_sst_offset_b0], v[v_sst_offset_b0], v[v_tmp]
    v_mov_b32 v[v_tmp], 0x8000
    v_xor_b32 v[v_sst_offset_b1], v[v_tmp], v[v_sst_offset_b0]
"""
        cta_n = self.tile.cta_n
        global_bk1 = self.tile.global_bk1
        log2_smem_b_k1 = int(math.log2(self.tile.smem_b_k1))
        log2_smem_bk0_stride = int(math.log2(cta_n * global_bk1))
        log2_compute_dt_size = int(math.log2(self.compute_datatype.data_size))
        a_smem_size = self.get_a_smem_size()
        sst_b_offset_src = SST_B_OFFSET.format(F_cta_n=cta_n, F_global_bk1=global_bk1, F_log2_smem_b_k1=log2_smem_b_k1, F_log2_smem_bk0_stride=log2_smem_bk0_stride, F_log2_compute_dt_size=log2_compute_dt_size, F_a_smem_size=a_smem_size)
        return sst_b_offset_src

    def gen_sld_a_offset(self):
        SLD_A_OFFSET = """
    ; load A to shared mem offset
    ; sld_iak0 = laneid / inst_m * ((block_m + pad) * ak1)
    ; sld_im = lane_id % inst_m + wave_im
    ; sld_offset_a = sld_im * ak1 + sld_iak0
    v_lshrrev_b32 v[v_sld_iak0], {F_log2_inst_m}, v[v_lane_id]
    v_mov_b32 v[v_tmp], {F_smem_ak0_stride}
    v_mul_lo_u32 v[v_sld_iak0], v[v_tmp], v[v_sld_iak0] 
    v_and_b32 v[v_sld_im], {F_inst_m_minus_1}, v[v_lane_id]
    v_add_lshl_u32 v[v_sld_im], v[v_sld_im], s[s_wave_im], {F_log2_smem_ak1}
    v_add_lshl_u32 v[v_sld_offset_a0], v[v_sld_iak0], v[v_sld_im], {F_log2_sizeof_dt}
    v_mov_b32 v[v_tmp], 0x8000
    v_xor_b32 v[v_sld_offset_a1], v[v_tmp], v[v_sld_offset_a0]
"""
        log2_inst_m = int(math.log2(self.tile.inst_m))
        smem_ak0_stride = (self.tile.cta_m + self.smem_a_padding) * self.tile.smem_a_k1
        inst_m_minus_1 = self.tile.inst_m - 1
        log2_smem_ak1 = int(math.log2(self.tile.smem_a_k1))
        log2_sizeof_dt = int(math.log2(self.a_datatype.data_size))
        sld_offfset_src = SLD_A_OFFSET.format(F_log2_inst_m=log2_inst_m, F_smem_ak0_stride=smem_ak0_stride, F_inst_m_minus_1=inst_m_minus_1, F_log2_smem_ak1=log2_smem_ak1, F_log2_sizeof_dt=log2_sizeof_dt)
        return sld_offfset_src

    def gen_sld_b_offset(self):
        SLD_B_OFFSET = """
    ; load B to shared mem offset
    ; k1 = max(ak1, bk1)
    ; sld_ibk0 = laneid / inst_n * (block_n * k1)
    ; sld_in = laneid % inst_n + wave_in
    ; sld_offset_b = sld_ibk0 + sld_in * bk1
    ; padding = sld_offset_b / 64 * 8
    ; sld_offset_b = padding + sld_offset_b
    v_lshrrev_b32 v[v_sld_ibk0], {F_log2_inst_n}, v[v_lane_id]
    v_lshlrev_b32 v[v_sld_ibk0], {F_log2_smem_bk0_stride}, v[v_sld_ibk0]
    v_and_b32 v[v_sld_in], {F_inst_n_minus_1}, v[v_lane_id]
    v_add_lshl_u32 v[v_sld_in], v[v_sld_in], s[s_wave_in], {F_log2_smem_bk1}
    v_add_u32 v[v_sld_offset_b0], v[v_sld_in], v[v_sld_ibk0]
    v_lshlrev_b32 v[v_sld_offset_b0], {F_log2_sizeof_dt}, v[v_sld_offset_b0]
    v_mov_b32 v[v_tmp], {F_a_smem_size}
    v_add_u32 v[v_sld_offset_b0], v[v_sld_offset_b0], v[v_tmp]
    v_mov_b32 v[v_tmp], 0x8000
    v_xor_b32 v[v_sld_offset_b1], v[v_tmp], v[v_sld_offset_b0]
"""
        log2_inst_n = int(math.log2(self.tile.inst_n))
        log2_smem_bk0_stride = int(math.log2(self.tile.cta_n * self.tile.smem_b_k1))
        inst_n_minus_1 = self.tile.inst_n - 1
        log2_smem_bk1 = int(math.log2(self.tile.smem_b_k1))
        log2_sizeof_dt = int(math.log2(self.compute_datatype.data_size))
        a_smem_size = self.get_a_smem_size()
        sld_b_offset_src = SLD_B_OFFSET.format(F_log2_inst_n=log2_inst_n, F_log2_smem_bk0_stride=log2_smem_bk0_stride, F_inst_n_minus_1=inst_n_minus_1, F_log2_smem_bk1=log2_smem_bk1, F_log2_sizeof_dt=log2_sizeof_dt, F_a_smem_size=a_smem_size)
        return sld_b_offset_src

    def gen_dup_scale_and_magic_num(self, wait_cnt_for_scale):
        DUP_SCALE_AND_MAGIC_NUM = """
    ; duplicate scale 
    s_waitcnt vmcnt({})
    v_mov_b32 v[v_scale + 1], v[v_scale + 0]

    ; v_pk_mul_f32 v[v_sub_magic_num + 0 : v_sub_magic_num + 1], v[v_scale + 0 : v_scale + 1], v[v_sub_magic_num + 0 : v_sub_magic_num + 1]
    v_mul_f32 v[v_sub_magic_num + 0], v[v_scale + 0], v[v_sub_magic_num + 0]
    v_mul_f32 v[v_sub_magic_num + 1], v[v_scale + 1], v[v_sub_magic_num + 1]
"""
        return DUP_SCALE_AND_MAGIC_NUM.format(wait_cnt_for_scale)

    def gen_clear_acc_vgpr(self, acc_num):
        CLEAR_ACC = """
    ; clear ACC vgpr
    .cnt = 0
    .rept {F_acc_num}
        v_mov_b32 v[v_c + .cnt], 0
        .cnt = .cnt + 1
    .endr
"""
        return CLEAR_ACC.format(F_acc_num=acc_num)

    def gen_pipeline(self):
        o_pipeline = self.pipeline.pipeline_select()
        return o_pipeline.k_pipeline_src

    def gen_write_out(self):
        

    def gen_kernel(self):
        # traits
        lds_size = self.get_lds_size()
        warp_size = self.get_warp_size()
        kernel_name = self.get_kernel_name()

        # macros
        kernel_str = ""
        m_print = common_macro.PrintMacro("print")
        m_dequant = common_macro.DequantMacro("dequant")
        m_mfma = common_macro.MfmaMacro("mfma", self.wg_repeat_m, self.wg_repeat_n)
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
        dict_sgprs = self.get_sgpr_dict()
        k_sgprs = sgprs.Sgprs(**dict_sgprs)
        kernel_str += k_sgprs.sgprs_body
        # print(kernel_str)

        # vgprs
        dict_vgprs = self.get_vgpr_dict()
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
            k_args.kargs_offset + 12,
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

        # scale load
        scale_load_str = self.gen_scale_load()
        kernel_str += scale_load_str

        # A gld address
        a_gld_addr_str = self.gen_a_matrix_gld_addr()
        kernel_str += a_gld_addr_str

        # A global prefetch
        a_gld_load_str = self.gen_a_matrix_gld_inst("v_gld_a0")
        kernel_str += a_gld_load_str

        # B gld address
        b_gld_addr_str = self.gen_b_matrix_gld_addr()
        kernel_str += b_gld_addr_str

        # B global prefetch
        b_gld_load_str = self.gen_b_matrix_gld_inst("v_gld_b0")
        kernel_str += b_gld_load_str

        # C global store address
        c_gst_addr_str = self.gen_c_gst_addr()
        kernel_str += c_gst_addr_str
   
        # A sst offset
        a_sst_addr_str = self.gen_sst_a_offset()
        kernel_str += a_sst_addr_str

        # B sst offset
        b_sst_addr_str = self.gen_sst_b_offset()
        kernel_str += b_sst_addr_str

        # A sld offset
        a_sld_addr_str = self.gen_sld_a_offset()
        kernel_str += a_sld_addr_str

        # B sld offset
        b_sld_addr_str = self.gen_sld_b_offset()
        kernel_str += b_sld_addr_str

        # dup scale and magic num
        waitcnt_scale = self.thread_vec_a[0] + self.thread_vec_b[0]
        dup_scale_m_num = self.gen_dup_scale_and_magic_num(waitcnt_scale)
        kernel_str += dup_scale_m_num

        # clear acc register
        clear_acc = self.gen_clear_acc_vgpr(self.acc_num)
        kernel_str += clear_acc

        # pipeline
        pipeline_src = self.gen_pipeline()
        kernel_str += pipeline_src

        # write out part
        write_out = c_write_out.WriteOut()
        kernel_str += write_out.c_write_out_src

        print(kernel_str)
        # program end
        p_end_str = self.gen_program_end() 
        kernel_str += p_end_str
        kernel_str += k_rodata
        kernel_str += k_amdgpu_metadata

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

    

