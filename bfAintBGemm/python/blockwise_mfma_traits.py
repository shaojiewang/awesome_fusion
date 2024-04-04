from dataclasses import dataclass

@dataclass
class BlockwiseMfmaTraits:
    cta_m: int
    cta_n: int
    cta_k: int
    cta_num_m: int
    cta_num_n: int
    cta_num_k: int
    wave_num_m: int
    wave_num_n: int
    inst_m: int
    inst_n: int
    inst_k: int
    
