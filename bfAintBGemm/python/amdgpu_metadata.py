
class AmdgpuMetadata(object):
    def __init__(self, name, sgpr_count, vgpr_count, karg_align, karg_size, lds_used, scratch_used, warp_size, cta_size, **kargs):
        self.name = name
        self.sgpr_count = sgpr_count
