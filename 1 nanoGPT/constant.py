import torch

n_head = 4  # 多头注意力的头数
n_layer = 4  # transformer层数
dropout = 0.0  # dropout率

context_size = 32
embd_dim = 64
batch_size = 16
lr = 1e-3
epochs = 5000
eval_epochs = 200
eval_interval = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
