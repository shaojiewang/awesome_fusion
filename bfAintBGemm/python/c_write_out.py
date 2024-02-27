class WriteOut(object):
    def __init__(self, repeat_m, repeat_n, num_warp_m, num_warp_n, inst_m, inst_n, warp_size, sizeof_dt):
        self.vgpr_per_group = 4
        self.repeat_m = repeat_m
        self.repeat_n = repeat_n
        self.num_warp_m = num_warp_m
        self.num_warp_n = num_warp_n
        self.inst_m = inst_m
        self.inst_n = inst_n
        self.warp_size = warp_size
        self.cta_size = num_warp_m * num_warp_n * warp_size
        self.cta_m = inst_m * repeat_m * num_warp_m
        self.cta_n = inst_n * repeat_n * num_warp_n

        self.sizeof_dt = sizeof_dt

        self.row_per_vgpr =  warp_size // inst_n
        self.acc_group_offset = self.cta_n * self.row_per_vgpr * self.vgpr_per_group * self.sizeof_dt
        self.acc_offset = self.cta_n * self.sizeof_dt
        self.repeat_n_offset = inst_n * num_warp_n * sizeof_dt

        self.c_write_out_src = self.gen_c_write_out()

    def gen_c_write_out(self):
        WRITE_OUT_LABEL = """
label_write_out_c:
    s_nop 15
    s_barrier
    ; rtz mode (not so accurate)
"""

        COMMENT = """
    ; store to lds
    ; within 1 inst group
    ; imm_offset = block_n * sizeof(datatype) * v_groups * n_per_vgpr * i_inst (64 * 2 * 4 * 2)
    ; within 1 vgpr group
    ; imm_offset = block_n * sizeof(datatype) * i_vgpr (64 * 2)
    ; imm_offset = block_n * sizeof(datatype) * v_groups * n_per_vgpr * i_inst + block_n * sizeof(datatype) * i_vgpr
"""

        CTA_BARRIER = """
    s_waitcnt lgkmcnt(0)
    s_barrier
"""

        SST_C = """
    .v_c_inst_cnt = 0
    .rept {F_vgpr_groups}
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + {F_acc_offst} + .v_c_inst_cnt * 4 + 0], offset: {F_repeat_n_offset} + {F_acc_group_offset} * .v_c_inst_cnt + {F_acc_offset} * 0
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + {F_acc_offst} + .v_c_inst_cnt * 4 + 1], offset: {F_repeat_n_offset} + {F_acc_group_offset} * .v_c_inst_cnt + {F_acc_offset} * 1
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + {F_acc_offst} + .v_c_inst_cnt * 4 + 2], offset: {F_repeat_n_offset} + {F_acc_group_offset} * .v_c_inst_cnt + {F_acc_offset} * 2
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + {F_acc_offst} + .v_c_inst_cnt * 4 + 3], offset: {F_repeat_n_offset} + {F_acc_group_offset} * .v_c_inst_cnt + {F_acc_offset} * 3
        .v_c_inst_cnt = .v_c_inst_cnt + 1
    .endr
"""

        SLD_COMMENT = """
    ; load from lds
    ; imm_offset = 16 * threadim.x * i
"""

        SLD_C = """
    ds_read_b128 v[v_c + {F_sld_c_offset} + 0 : v_c + {F_sld_c_offset} + 3], v[v_sld_offset_c], offset: 16 * 256 * {F_sld_c_idx}
"""

        GST_BEGIN = """
    ; store res to global
    v_cmpx_eq_u32 vcc, 1, v[v_c_n_flag]    
"""

        GST_C = """
    s_mul_i32 s[s_tmp], {F_rows_per_gst}, s[s_ldc]
    buffer_store_dwordx4 v[v_c + {F_gst_c_offset} + 0 : v_c + {F_gst_c_offset} + 3], v[v_gst_offset_c], s[s_ptr_c + 0 : s_ptr_c + 3], s[s_tmp] offen offset: 0
"""
        GST_END = """
    s_mov_b64 exec, -1
"""

        write_out_src = WRITE_OUT_LABEL + COMMENT
        vgpr_groups = self.inst_m * self.inst_n // (self.vgpr_per_group * self.warp_size)
        vgpr_per_inst = vgpr_groups * self.vgpr_per_group
        for m in range(self.repeat_m):
            for n in range(self.repeat_n):
                write_out_src += SST_C.format(F_vgpr_groups=self.vgpr_per_group, F_acc_offst=vgpr_per_inst * (m * self.repeat_n + n), F_repeat_n_offset=self.repeat_n_offset * n, F_acc_group_offset=self.acc_group_offset, F_acc_offset=self.acc_offset)

            write_out_src += CTA_BARRIER
            sld_c_num = self.inst_m * self.inst_n * self.num_warp_m * self.num_warp_n * self.repeat_n // (self.cta_size * (16 // self.sizeof_dt))
            write_out_src += SLD_COMMENT
            for i in range(sld_c_num):
                write_out_src += SLD_C.format(F_sld_c_offset=i * 4, F_sld_c_idx=i)
            write_out_src += CTA_BARRIER

            write_out_src += GST_BEGIN
            rows_per_sld = (self.cta_size * (16 // self.sizeof_dt)) // self.cta_n
            for i in range(sld_c_num):
                write_out_src += GST_C.format(F_rows_per_gst=rows_per_sld * i + self.num_warp_m * self.inst_m * m, F_gst_c_offset=i * 4)

            write_out_src += GST_END
            
        return write_out_src

