#checks if gpu can save and load data
import torch
t = torch.randn(2,2)
r = t.to(0)
print(r)
