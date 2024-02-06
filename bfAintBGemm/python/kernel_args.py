from dataclasses import dataclass

@dataclass
class KernelArgs(object):
    def __init__(self, **kernel_args):
        self.karg_begin_byte = 0
        self.karg_ele_bytes = 4
        self.kargs_body = self.write_kargs(**kernel_args)

    def write_kargs(self, **kernel_args):
        KARG = """.set {}, {}\n"""
        KARGS = """;kernel arguments OFFSET, shift in 1 byte\n"""
        for key in kernel_args.keys():
            KARGS += KARG.format(key, self.karg_begin_byte)
            self.karg_begin_byte += self.karg_ele_bytes * kernel_args[key]
        return KARGS
        
