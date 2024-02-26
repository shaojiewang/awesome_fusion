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

        SST_C = """
    .v_c_inst_cnt = 0
    .rept {F_vgpr_groups}
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 0], offset: {F_repeat_n_offset} + {F_acc_group_offset} * .v_c_inst_cnt + {F_acc_offset} * 0
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 1], offset: {F_repeat_n_offset} + {F_acc_group_offset} * .v_c_inst_cnt + {F_acc_offset} * 1
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 2], offset: {F_repeat_n_offset} + {F_acc_group_offset} * .v_c_inst_cnt + {F_acc_offset} * 2
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 3], offset: {F_repeat_n_offset} + {F_acc_group_offset} * .v_c_inst_cnt + {F_acc_offset} * 3
        .v_c_inst_cnt = .v_c_inst_cnt + 1
    .endr
    s_waitcnt lgkmcnt(0)
    s_barrier
"""

        SLD_C = """
    ; load from lds
    ; imm_offset = 16 * threadim.x * i
    ds_read_b128 v[v_c + 0 : v_c + 3], v[v_sld_offset_c], offset: 16 * 256 * 0
    ds_read_b128 v[v_c + 4 : v_c + 7], v[v_sld_offset_c], offset: 16 * 256 * 1

    s_mov_b32 s[s_tmp], 0
    s_waitcnt lgkmcnt(0)
    s_barrier

"""

        GST_C = """
    ; store res to global
    v_cmpx_eq_u32 vcc, 1, v[v_c_n_flag]    
    buffer_store_dwordx4 v[v_c + 0 : v_c + 3], v[v_gst_offset_c], s[s_ptr_c + 0 : s_ptr_c + 3], s[s_tmp] offen offset: 0
    s_mul_i32 s[s_tmp], 16, s[s_ldc]
    buffer_store_dwordx4 v[v_c + 4 : v_c + 7], v[v_gst_offset_c], s[s_ptr_c + 0 : s_ptr_c + 3], s[s_tmp] offen offset: 0
    s_mov_b64 exec, -1
"""

        write_out_src = ""
        vgpr_groups = self.inst_m * self.inst_n // (self.vgpr_per_group * self.warp_size)
        return WRITE_OUT

