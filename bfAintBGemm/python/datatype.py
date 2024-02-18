from dataclasses import dataclass

@dataclass
class DataType(object):
    data_type : str
    data_size : int
    subbyte_bits : int

F16 = DataType("float16", 2, 16)
BF16 = DataType("bfloat16", 2, 16)
F32 = DataType("float32", 4, 32)
I8 = DataType("int8", 1, 8)

