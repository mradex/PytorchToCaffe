import torch
import torch.nn as nn

class SplitFCLayer(nn.Module):
    def __init__(self):
        super(SplitFCLayer, self).__init__()
        self.fc = nn.Linear(3, 4)
        with torch.no_grad(): 
            self.fc.weight.fill_(1)
            self.fc.bias.fill_(0)
        
    def forward(self, x):
        x1, x2 = torch.split(x, 3)
        y1 = self.fc(x1)
        y2 = self.fc(x2)
        y = torch.cat((y1,y2), 0)
        return y

model = SplitFCLayer()

input = torch.ones(6, 3)
output = model(input)

torch.save(input, "input.pth")
torch.save(model.state_dict(), "SplitFCLayer.pth")

print(output)
