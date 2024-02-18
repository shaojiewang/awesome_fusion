from dataclasses import dataclass

@dataclass
class TextSeg(object):
    name : str
    p2align : int

    def __post_init__(self):
        k_text_seg = self.gen_text_seg()
        self.text_seg_str = k_text_seg.format(self.name, self.p2align, self.name)

    def gen_text_seg(self):
        TEXTSEG = """
.text
.global {}
.p2align {}
.type {},@function
"""
        return TEXTSEG

