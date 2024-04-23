def check_new_col(row, col):
    res = ''
    for i in range(row):
        for j in range(col):
            j_h = j // 2
            i_xor_j_h = i ^ j_h
            res_num = i_xor_j_h * 2 + j % 2
            res += str(res_num) + ", "
        res += '\n'
    return res
  
row = 8
col = 32
print(check_new_col(row, col))

