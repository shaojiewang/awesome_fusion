from dataclasses import dataclass

@dataclass
class Rodata(object):
    name : str
    lds_size : int
    sgpr_dispatch_ptr : int
    sgpr_karg_seg_ptr : int
    sgpr_cta_idx : int
    sgpr_cta_idy : int
    sgpr_cta_idz : int
    vgpr_tidx : int
    next_free_vgpr : int
    next_free_sgpr : int
    ieee_mode : int
    dx10_clamp : int
    accum_offset : int

    def __post_init__(self):
        k_rodata_str = self.gen_rodata()
        self.rodata_str = k_rodata_str.format(
            self.name,
            self.lds_size,
            self.sgpr_dispatch_ptr,
            self.sgpr_karg_seg_ptr,
            self.sgpr_cta_idx,
            self.sgpr_cta_idy,
            self.sgpr_cta_idz,
            self.vgpr_tidx,
            self.next_free_vgpr,
            self.next_free_sgpr,
            self.ieee_mode,
            self.dx10_clamp,
            self.accum_offset)

    def gen_rodata(self):
        RODATA = """
.rodata
.p2align 6
.amdhsa_kernel {}
    .amdhsa_group_segment_fixed_size {}
    .amdhsa_user_sgpr_dispatch_ptr {}
    .amdhsa_user_sgpr_kernarg_segment_ptr {}
    .amdhsa_system_sgpr_workgroup_id_x {}
    .amdhsa_system_sgpr_workgroup_id_y {}
    .amdhsa_system_sgpr_workgroup_id_z {}
    .amdhsa_system_vgpr_workitem_id {}
    .amdhsa_next_free_vgpr {}
    .amdhsa_next_free_sgpr {}
    .amdhsa_ieee_mode {}
    .amdhsa_dx10_clamp {}
    .amdhsa_accum_offset {}
    # .amdhsa_wavefront_size32 1
    # .amdhsa_workgroup_processor_mode 0
.end_amdhsa_kernel
"""
        return RODATA

