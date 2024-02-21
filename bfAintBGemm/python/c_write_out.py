class WriteOut(object):
    def __init__(self):
        self.c_write_out_src = self.gen_c_write_out()

    def gen_c_write_out(self):
        WRITE_OUT = """
label_write_out_c:
    s_nop 15
    s_barrier
    ; rtz mode (not so accurate)

    ; store to lds
    ; within 1 inst group
    ; imm_offset = block_n * sizeof(datatype) * v_groups * n_per_vgpr * i_inst (64 * 2 * 4 * 2)
    ; within 1 vgpr group
    ; imm_offset = block_n * sizeof(datatype) * i_vgpr (64 * 2)
    ; imm_offset = block_n * sizeof(datatype) * v_groups * n_per_vgpr * i_inst + block_n * sizeof(datatype) * i_vgpr
    .v_c_inst_cnt = 0
    .rept 4
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 0], offset: 128 * 2 * 4 * 2 * .v_c_inst_cnt + 128 * 2 * 0
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 1], offset: 128 * 2 * 4 * 2 * .v_c_inst_cnt + 128 * 2 * 1
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 2], offset: 128 * 2 * 4 * 2 * .v_c_inst_cnt + 128 * 2 * 2
        ds_write_b16_d16_hi v[v_sst_offset_c], v[v_c + .v_c_inst_cnt * 4 + 3], offset: 128 * 2 * 4 * 2 * .v_c_inst_cnt + 128 * 2 * 3
        .v_c_inst_cnt = .v_c_inst_cnt + 1
    .endr
    s_waitcnt lgkmcnt(0)
    s_barrier

    ; load from lds
    ; imm_offset = 16 * threadim.x * i
    ds_read_b128 v[v_c + 0 : v_c + 3], v[v_sld_offset_c], offset: 16 * 256 * 0
    ds_read_b128 v[v_c + 4 : v_c + 7], v[v_sld_offset_c], offset: 16 * 256 * 1

    s_mov_b32 s[s_tmp], 0
    s_waitcnt lgkmcnt(0)
    s_barrier

    ; store res to global
    v_cmpx_eq_u32 vcc, 1, v[v_c_n_flag]    
    buffer_store_dwordx4 v[v_c + 0 : v_c + 3], v[v_gst_offset_c], s[s_ptr_c + 0 : s_ptr_c + 3], s[s_tmp] offen offset: 0
    s_mul_i32 s[s_tmp], 16, s[s_ldc]
    buffer_store_dwordx4 v[v_c + 4 : v_c + 7], v[v_gst_offset_c], s[s_ptr_c + 0 : s_ptr_c + 3], s[s_tmp] offen offset: 0
    s_mov_b64 exec, -1
"""
        return WRITE_OUT

