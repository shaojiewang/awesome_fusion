from dataclasses import dataclass

@dataclass
class KernelArgTraits:
    size : int
    value_kind : str
    value_type : str
    address_space : str
    is_const : bool 

class KernelArgs(object):
    def __init__(self, **kernel_args):
        self.karg_begin_byte = 0
        self.kargs_body = self.write_kargs(**kernel_args)

    def write_kargs(self, **kernel_args):
        KARG = """.set {}, {}\n"""
        KARGS = """;kernel arguments OFFSET, shift in 1 byte\n"""
        for key in kernel_args.keys():
            KARGS += KARG.format(key, self.karg_begin_byte)
            self.karg_begin_byte += kernel_args[key].size
        return KARGS
        
