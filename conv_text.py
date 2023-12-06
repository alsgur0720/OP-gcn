import torch.nn as nn
import torch


# With square kernels and equal stride
# non-square kernels and unequal stride and with padding
m1 = nn.Conv3d(64, 128, (5, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
m2 = nn.Conv3d(64, 128, (2, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0))
input1 = torch.randn(128, 64, 5, 64, 25)
input2 = torch.randn(128, 64, 5, 64, 25)
output1 = m1(input1)
output2 = m2(input2)


output = torch.cat([output1, output2],  dim = 2)
print(output.size())