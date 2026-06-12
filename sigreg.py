import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

def w(t, lmbda=1):
    return torch.exp(-t**2/(2*lmbda**2))

def SIGReg(Z, M=1024, T=17):
    B,D = Z.shape
    u = F.normalize(torch.randn(D,M, device=Z.device), dim=0)
    projs = Z@u
    x = torch.linspace(0.2, 4, T, device=Z.device)
    x = x.view(1,1,-1)
    projs = projs.unsqueeze(-1)
    phi_N_real = lambda t: torch.cos(t*projs).mean(dim=0)
    phi_N_im = lambda t: torch.sin(t*projs).mean(dim=0)
    phi_0 = lambda t: torch.exp(-0.5 * (t**2)) 
    y = w(x)*((phi_N_real(x) - phi_0(x))**2+(phi_N_im(x))**2)
    return torch.trapezoid(y, x, dim=-1).mean()