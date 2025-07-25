import torch
import torch.nn as nn

class SimpleConvLayer(nn.Module):
    def __init__(self):
        super(SimpleConvLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(3,3), bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1) 

    def forward(self, x):
        return self.conv(x)

model = SimpleConvLayer()

input_tensor = torch.ones(1, 3, 4, 4)

output = model(input_tensor)
print("Output shape:", output.shape)
print(output)

torch.save(model.state_dict(), "convlayer.pth")
