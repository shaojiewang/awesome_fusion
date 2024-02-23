import math

def is_power_of_2(n):
    return n > 0 and (n & (n -1 )) == 0

def log2_int(data_in) -> int :
    if is_power_of_2(data_in):
        return int(math.log2(data_in))
    else:
        assert falsh, "{} is not power of 2".format(data_in)
