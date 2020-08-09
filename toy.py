
import os
import sys
import time
import numpy as np
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.rand(10)
x = x.cuda()
#print(2*x)

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
