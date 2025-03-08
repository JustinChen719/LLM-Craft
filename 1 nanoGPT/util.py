import torch
from constant import *

with open('../data/tinyshakespeare.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("数据集中的总字符数: ", len(text))
# 文本中出现的所有的字符
chars = sorted(list(set(text)))
vocab_size = len(chars)
print("文本中出现的字符: ", ''.join(chars))
print("文本中出现的字符数量: ", vocab_size)

# 简单的tokenizer实现
# 这是一个简单的tokenizer实现，就是简单的把文本中出现过的所有字符按顺序给予一个整数0~vocab_size-1
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

# 简单的编码和解码规则
# 编码：给定一个字符串输出一个list的整数
encode = lambda s: [stoi[c] for c in s]
# 解码：给定一个list的整数输出一个字符串
decode = lambda l: ''.join([itos[i] for i in l])

## 数据集划分以及数据分批
data = torch.tensor(encode(text), dtype=torch.long)
# 90%划分训练集和验证集
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # 生成一个batch的数据，x为输入，y是target
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    # y相对于x移位一个token，因为训练的目标就是预测下一个token
    x = torch.stack([data[i:i + context_size] for i in ix])
    y = torch.stack([data[i + 1:i + context_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_epochs)
        for it in range(eval_epochs):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[it] = loss
        out[split] = losses.mean()
    return out
