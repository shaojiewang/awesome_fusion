from dataclasses import dataclass
from typing import Callable
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

    inst_gld_a: Callable
    inst_gld_b: Callable
    inst_dequant_a: Callable
    inst_dequant_b: Callable
    inst_ds_write_a: Callable
    inst_ds_write_b: Callable
    inst_ds_read_a: Callable
    inst_ds_read_b: Callable
    inst_mfma: Callable

    blockwise_mfma_traits: BlockwiseMfmaTraits

    double_buffer_lds: bool
    wavelet_mode: bool
    
