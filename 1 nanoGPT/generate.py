import torch
from model import GPT

model_path = "./output/nanoGPT_5000_epochs.pt"
model = GPT(65, 4, 4, 0)
model.load_state_dict(torch.load(model_path))
model.eval()
model = model.to("cuda")

input = torch.zeros((1, 1), dtype=torch.long).to("cuda")
# print(decode(model.generate(input, 1000)[0].tolist()))

model.generate(input, 1000)
