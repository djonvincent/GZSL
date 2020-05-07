import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, d_in, d_out):
        super(Classifier, self).__init__()
        self.layer = nn.Linear(d_in, d_in)
        self.layer2 = nn.Linear(d_in, d_out)

    def f1(self, x):
        x = self.layer(x)
        return F.relu(x)

    def forward(self, x):
        x = self.f1(x)
        return self.layer2(x)

class Compatibility(nn.Module):
    def __init__(self, d_in, d_out):
        super(Compatibility, self).__init__()
        self.layer = nn.Linear(d_in, d_out)

    def forward(self, x, s):
        x = self.layer(x)
        return F.linear(x, s)

class SegregationNetwork(nn.Module):
    def __init__(self, nclasses):
        super(SegregationNetwork, self).__init__()

        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(1536, nclasses)

    def forward(self, xi, xj, xk):
        xi = F.relu(self.fc1(xi))
        xj = F.relu(self.fc1(xj))
        xk = F.relu(self.fc1(xk))

        xijk = torch.cat((xi,xj,xk), dim=1)
        return torch.sigmoid(self.fc2(xijk))

def constituency_loss(u, beta, g=1):
    nm = beta == 0
    m = beta > 0
    return torch.mean(u[nm]**2) + g*torch.mean((u[m] - beta[m])**2)
    #return torch.sum(u[nm]**2) + g*torch.sum((u[m] - beta[m])**2)

