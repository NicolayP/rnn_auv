import torch
#torch.autograd.set_detect_anomaly(True)
import pypose as pp
import numpy as np
#from tqdm import tqdm
from utile import tdtype, npdtype, to_euler, gen_imgs_3D
import os

@torch.jit.ignore
def broadcast_inputs(x, y):
    """ Automatic broadcasting of missing dimensions """
    if y is None:
        xs, xd = x.shape[:-1], x.shape[-1]
        return (x.reshape(-1, xd).contiguous(), ), x.shape[:-1]
    out_shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    shape = out_shape if out_shape != torch.Size([]) else (1,)
    x = x.expand(shape+(x.shape[-1],)).reshape(-1,x.shape[-1]).contiguous()
    y = y.expand(shape+(y.shape[-1],)).reshape(-1,y.shape[-1]).contiguous()

    return x, y, torch.Tensor(list(out_shape)).to(torch.int32)


class SE3:
    @staticmethod
    def Exp(x):
        """
        Exponential map for se3
        """
        return pp.se3_Exp.apply(x)
    
    @staticmethod
    def rotation(x):
        return x[..., 3:7]
    
    @staticmethod
    def Inv(X):
        return pp.SE3_Inv.apply(X)
    
    @staticmethod
    def Adj(X, a):
        input_a, input_b, out_shape = broadcast_inputs(X, a)
        out = pp.SE3_AdjXa.apply(input_a, input_b)
        dim = -1 
        if out.numel() == 0:
            dim = X.shape[-1]
        test = to_1d_int_list(out_shape) + [dim]
        return out.view(test)
    
def to_1d_int_list(x):
    result: List[int] = []
    for i in x:
        result.append(i.item())
    return result


class SO3:    
    @staticmethod
    def Act(X, p):
        input_a, input_b, out_shape = broadcast_inputs(X, p)
        if p.shape[-1]==3:
            out = pp.SO3_Act.apply(input_a, input_b)
        else:
            out = pp.SO3_Act4.apply(input_a, input_b)
        dim = -1 
        if out.numel() == 0:
            dim = X.shape[-1]
        test = to_1d_int_list(out_shape) + [dim]
        return  out.view(test)
    
    @staticmethod
    def matrix(input):
        """ To 3x3 matrix """
        I = torch.eye(3, dtype=input.dtype, device=input.device)
        I = I.view([1] * (input.dim() - 1) + [3, 3])
        unsq = input.unsqueeze(-2)
        acted = SO3.Act(unsq, I).transpose(-1,-2)
        return acted

class se3:
    @staticmethod
    def Exp(x):
        """
        Exponential map for se3
        """
        return pp.se3_Exp.apply(x)