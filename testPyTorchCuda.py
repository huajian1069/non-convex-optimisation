import torch; print("torch version: ",torch.__version__); print("cuda_is_available: ",torch.cuda.is_available());print("cuda version: ", torch.version.cuda);x=torch.rand(10).cuda();print(x)
