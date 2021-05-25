import torch
import torch.nn as nn
import torch.nn.functional as F

def normalize(K_ij):
    K_ij -= K_ij.min(1, keepdim=True)[0]
    K_ij /= K_ij.max(1, keepdim=True)[0]
    return K_ij

class RBF(nn.Module):
    """
    RBF Kernel:
    K_ij = exp(-|X_i-X_j|^2/(2\sigma^2))
    """
    def __init__(self, sigma=1.0):
        super(RBF, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        

    def forward(self, x, y):
        if len(x.size()) > 2:
            x = x.contiguous().view(x.size(0), -1)
            y = y.contiguous().view(y.size(0), -1)

        x_i = x.unsqueeze(1)    # B1,1,D
        y_j = y.unsqueeze(0)    # 1,B2,D

        sqd = torch.sum(torch.pow(x_i-y_j, 2), dim=2)
        K_ij = torch.exp(-0.5*sqd/torch.pow(self.sigma, 2))  # D1, D2
    

        return K_ij
        
class Poly(nn.Module):
    """
    K_ij = <X_i, X_j>^d
    """
    def __init__(self, d=2):
        super(Poly, self).__init__()
        assert d in [1,2,3]
        self.d = nn.Parameter(torch.tensor(d), requires_grad=False)

    def forward(self, x, y):
        if len(x.size()) > 2:
            x = x.contiguous().view(x.size(0), -1)
            y = y.contiguous().view(y.size(0), -1)


        K_ij = x.matmul(y.t()).pow(self.d)# N1, N2
        K_ij = normalize(K_ij)
        

        return K_ij
    

class Cosine(nn.Module):
    """
    K_ij = <X_i, X_j>/|X_i||X_j|
    """
    def __init__(self):
        super(Cosine, self).__init__()
        

    def forward(self, x, y):
        if len(x.size()) > 2:
            x = x.contiguous().view(x.size(0), -1)
            y = y.contiguous().view(y.size(0), -1)

        K_ij = x.matmul(y.t())
        norm = torch.norm(x, p=2)*torch.norm(y, p=2)
        K_ij /= norm


        return K_ij

class Sigmoid(nn.Module):
    """
    K_ij = tanh(\alpha<X_i, X_j>+c)
    """
    def __init__(self, alpha=1.0, c=0.0):
        super(Sigmoid, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.c = nn.Parameter(torch.tensor(c), requires_grad=False)
        

    def forward(self, x, y):
        if len(x.size()) > 2:
            x = x.contiguous().view(x.size(0), -1)
            y = y.contiguous().view(y.size(0), -1)


        K_ij = torch.tanh(self.alpha*x.matmul(y.t())+self.c)

        

        return K_ij
