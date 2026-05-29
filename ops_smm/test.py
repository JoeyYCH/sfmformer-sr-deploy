import os
from torch.utils.cpp_extension import CUDA_HOME
print(CUDA_HOME)

import torch
print(torch.__version__)
# 預期輸出應該包含 +cu128，例如: '2.6.0+cu128'

print(torch.version.cuda)
# 預期輸出: '12.8'

print(torch.cuda.is_available())
# 預期輸出: True