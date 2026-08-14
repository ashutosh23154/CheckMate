import torch
import math

def positional_encoding(seq_len, embed_dim):
    pe = torch.zeros(seq_len, embed_dim)
    for pos in range(seq_len):
        for i in range(0, embed_dim, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (i / embed_dim)))
            if i + 1 < embed_dim:
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / embed_dim)))
    return pe
