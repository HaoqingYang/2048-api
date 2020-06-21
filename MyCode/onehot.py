import numpy as np

CAND = 16
MAP_TABLE = {2**i: i for i in range(1,CAND)}
MAP_TABLE[0] = 0

def change_to_onehot(board):
    ret = np.zeros(shape=(4,4,CAND),dtype=float)
    for r in range(4):
        for c in range(4):
            ret[r,c,MAP_TABLE[board[r,c]]] = 1
    return ret


