
from dataclasses import dataclass

@dataclass
class Vgprs(object):
    def __init__(self, **vgprs):
        self.vgpr_offset = 0
        self.vgpr_ele = 1
        self.vgprs_body = self.write_vgprs(**vgprs)

    def write_vgprs(self, **vgprs):
        VGPR = """.set {}, {}\n"""
        VGPRS = """\n;vgpr\n"""
        for key in vgprs.keys():
            ele_off = self.vgpr_ele * vgprs[key]
            ele_align = min(ele_off, 4)
            self.vgpr_offset = (self.vgpr_offset + ele_align - 1) // ele_align * ele_align 
            VGPRS += VGPR.format(key, self.vgpr_offset)
            self.vgpr_offset += ele_off
        return VGPRS
        
