import torch
import torch.nn as nn
from convlayer import SimpleConvLayer

if __name__=='__main__':
    name='convlayer'
    net = SimpleConvLayer()
    net.load_state_dict(torch.load("convlayer.pth"))
    
    input=torch.load("input.pth")
    output = net(input)
    print("Output shape:", output.shape)
    print(output)
