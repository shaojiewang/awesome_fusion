from dataclasses import dataclass

@dataclass
class PipelineTraits:
    v_gld_a0: str
    v_gld_a1: str
    v_gld_b0: str
    v_gld_b1: str
    v_offset_a: str
    v_offset_b: str
    s_bs_a: str
    s_bs_b: str

    inst_gld_a: str
    inst_gld_b: str
    inst_ds_write_a: str
    inst_ds_write_b: str
    inst_ds_read_a: str
    inst_ds_read_b: str
    inst_mfma: str


