import torch
import torch.nn as nn
import numpy as np

class SplitConcat(nn.Module):
    def __init__(self):
        super(SplitConcat, self).__init__()
        self.fc1 = nn.Linear(16, 16)
        with torch.no_grad(): 
            self.fc1.weight.fill_(1)
            self.fc1.bias.fill_(0)
        
        self.fc2 = nn.Linear(16, 16)
        with torch.no_grad(): 
            self.fc2.weight.fill_(2)
            self.fc2.bias.fill_(0)

    def forward(self, x):
        x1, x2 = torch.split(x, 16, 1)
        y1 = self.fc1(x1)
        y2 = self.fc2(x2)
        #return y1
        y = torch.cat((y1,y2), 1)
        return y

model = SplitConcat()

input = torch.ones(1, 32)
#input = torch.ones(1, 16)
output = model(input)

torch.save(input, "input.pth")
np.save("input.npy", input.numpy())
torch.save(model.state_dict(), "model.pth")

print(output)
