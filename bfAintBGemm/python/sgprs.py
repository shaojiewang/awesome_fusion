
from dataclasses import dataclass

@dataclass
class Sgprs(object):
    def __init__(self, **sgprs):
        self.sgpr_offset = 0
        self.sgpr_ele = 1
        self.sgprs_body = self.write_sgprs(**sgprs)

    def write_sgprs(self, **sgprs):
        SGPR = """.set {}, {}\n"""
        SGPRS = """\n;sgpr\n"""
        for key in sgprs.keys():
            ele_off = self.sgpr_ele * sgprs[key]
            self.sgpr_offset = (self.sgpr_offset + ele_off - 1) // ele_off * ele_off
            SGPRS += SGPR.format(key, self.sgpr_offset)
            self.sgpr_offset += ele_off
        return SGPRS
        
