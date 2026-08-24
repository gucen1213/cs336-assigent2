import torch

row = torch.randint(0,10,(0,3,4))
print(row)
weight = torch.randint(0,10,(4,))
print(weight)

out = row * weight
print(out)

print(torch.sum(out, axis=-1))

print(row.shape[:-1])