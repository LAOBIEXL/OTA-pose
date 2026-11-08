import torch

import torch

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# 在新维度 dim=0 堆叠 -> 形状 [2, 3]
x = torch.stack([a, b], dim=0)  # tensor([[1,2,3],[4,5,6]])

# 在新维度 dim=1 堆叠 -> 形状 [3, 2]
y = torch.stack([a, b], dim=1)  # tensor([[1,4],[2,5],[3,6]])

print(x.shape)
print(y.shape)

print("!!!")