from dataclasses import dataclass

@dataclass
class GemmTileSize:
    cta_size : int # thread block size
    cta_m : int
    cta_n : int
    cta_k : int
    global_bk1 : int
    warp_m : int
    warp_n : int
    inst_m : int
    inst_n : int
    inst_k : int
    inst_blocks : int
    gmem_vec_a : int
    gmem_vec_b : int
    gmem_vec_c : int
    gmem_vec_scale : int
    smem_vec_a : int
    smem_vec_b : int
    smem_vec_c : int
    smem_a_k1 : int
    smem_b_k1 : int
    
@dataclass
class GemmKernelTraits:
    a_layout : str
    b_layout : str
    c_layout : str
    a_interleave : int
    b_interleave : int
    c_interleave : int
    a_datatype : str
    b_datatype : str
    c_datatype : str
    scale_datatype : str
    splitk : int
    tile : GemmTileSize
    pipeline : str

