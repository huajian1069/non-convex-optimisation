import torch
x = torch.rand(10)
x = x.cuda()
print(2*x)

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)