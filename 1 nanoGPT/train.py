import torch

from model import GPT
from util import vocab_size, get_batch, estimate_loss
from constant import *
import matplotlib.pyplot as plt

model = GPT(vocab_size, n_layer, n_head, dropout)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
train_losses = []
val_losses = []
for epoch in range(epochs):
    # 训练
    model.train()
    x, y = get_batch("train")
    logits, loss = model(x, y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # 评估
    if epoch % eval_interval == 0 or epoch == epochs - 1:
        losses = estimate_loss(model)
        print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        train_losses.append(losses["train"])
        val_losses.append(losses["val"])

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("loss.png")
plt.show()

# 保存模型
torch.save(model.state_dict(), "output/nanoGPT_5000_epochs.pt")
