from dataclasses import dataclass
from blockwise_mfma_traits import BlockwiseMfmaTraits

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

    inst_gld_a: list
    inst_gld_b: list
    inst_dequant_a: list
    inst_dequant_b: list
    inst_ds_write_a: list
    inst_ds_write_b: list
    inst_ds_read_a: list
    inst_ds_read_b: list
    inst_mfma: list

    blockwise_mfma_traits: BlockwiseMfmaTraits

    double_buffer_lds: bool
    wavelet_mode: bool
    
