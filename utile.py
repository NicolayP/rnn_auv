import torch
from torch.nn.functional import normalize
torch.autograd.set_detect_anomaly(True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# SE2
class ToSE2Mat(torch.nn.Module):
    def __init__(self):
        super(ToSE2Mat, self).__init__()
        self.se2exp = SE2Exp()

    def forward(self, s):
        '''
            transforms a s \in R^{3} into a Lie Element M \in R^{3*3}
            inputs:
            -------
                - s, the state in the R^{k*3} (x, y, yaw). Shape [k, 3]
            outputs:
            --------
                - M, a element of the Lie Group. Shape [k, 3, 3]
        '''
        return self.se2exp(s)


class Skew2(torch.nn.Module):
    def __init__(self):
        super(Skew2, self).__init__()
        const = torch.Tensor([[[0., -1.], [1., 0.]]])
        self.register_buffer('const_mul', const)

    def forward(self, yaw):
        return yaw * self.const_mul


class SO2Exp(torch.nn.Module):
    def __init__(self):
        super(SO2Exp, self).__init__()

    def forward(self, yaw):
        cos = torch.cos(yaw[..., None])
        sin = torch.sin(yaw[..., None])

        r0 = torch.concat([cos, -sin], dim=-1)
        r1 = torch.concat([sin, cos], dim=-1)
        r = torch.concat([r0, r1], dim=-2)
        return r


class SO2Log(torch.nn.Module):
    def __init__(self):
        super(SO2Log, self).__init__()
        pass

    def forward(self, M):
        sin = M[:, 1, 0]
        cos = M[:, 0, 0]
        yaw = torch.atan2(sin, cos)[..., None]
        return yaw


class SE2Exp(torch.nn.Module):
    def __init__(self):
        super(SE2Exp, self).__init__()
        pad = torch.Tensor([[[0., 0., 1.]]])
        self.register_buffer('pad_const', pad)
        self.v = SE2V()
        self.so2exp = SO2Exp()

    def forward(self, s):
        k = s.shape[0]
        #print("\t", "-"*5, "SE2 Exp", "-"*5)
        yaw = s[:, -1:] # shape [k, 1]
        rho = s[:, :2, None] # shape[k, 2, 1]

        #print("\tyaw: ", yaw.shape)
        #print("\trho: ", rho.shape)
        r = self.so2exp(yaw)
        #print("\tr:   ", r.shape)

        v = self.v(yaw) @ rho
        #print("\tv:   ", v.shape)

        row = torch.concat([r, v], dim=-1)
        #print("\trow: ", row.shape)
        pad = self.pad_const.broadcast_to((k, 1, 3))
        res = torch.concat([row, pad], dim=-2)
        #print("\tres: ", res.shape)
        return res


class SE2Log(torch.nn.Module):
    def __init__(self):
        super(SE2Log, self).__init__()
        self.so2log = SO2Log()
        self.se2v = SE2V()
        pass

    def forward(self, M):
        yaw = self.so2log(M)

        t = M[:, :2, 2:]
        vTheta = self.se2v(yaw)
        invV = torch.linalg.inv(vTheta)
        # remove last dim that was necessary to perform multiplication.
        rho = (invV @ t)[..., 0]
        return torch.concat([rho, yaw], dim=-1)[:, None]


class SE2V(torch.nn.Module):
    def __init__(self):
        super(SE2V, self).__init__()
        self.eps = 1e-10
        skew = torch.Tensor([[[0., -1.], [1., 0.]]])
        self.register_buffer("skew", skew)
        a = torch.eye(2)[None, ...]
        self.register_buffer("a", a)

    def forward(self, yaw):
        '''
            Computes V(\theta) = \frac{sin(\theta)}{\theta}*\boldsymbol{I} + \frac{1-cos(\theta)}{\theta}

            inputs:
            -------
                - \theta. Angle in radians. Shape [k, 1]
            
            outputs:
            --------
                V(\theta)
        '''
        #print("\n\t\t", "*"*5, "SE2V", "*"*5)
        k = yaw.shape[0]
        result = torch.zeros((k, 2, 2)).to(yaw.device)
        non_zero_theta = yaw[yaw > self.eps]
        #print("\t\tyaw:  ", yaw.shape)
        #print("\t\tno-0: ", non_zero_theta.shape)
        # when theta = 0 the first term is 1
        # and second is 0 (lim theta->0)
        if non_zero_theta.shape[0] > 0:
            tmp1 = torch.sin(non_zero_theta)/non_zero_theta
            tmp2 = (1 - torch.cos(non_zero_theta))/non_zero_theta
            tmp = tmp1[..., None, None]*self.a + tmp2[..., None, None]*self.skew
            result[(yaw > self.eps)[:, 0]] = tmp

        result[(yaw <= self.eps)[:, 0]] = self.a
        return result


class SE2Int(torch.nn.Module):
    def __init__(self):
        super(SE2Int, self).__init__()
        self.se2exp = SE2Exp()

    def forward(self, M, tau):
        '''
            Applies the perturbation Tau on M (in SE(2)) using the exponential mapping and the right plus operator.
            input:
            ------
                - M Element of SE(2), shape [k, 3, 3] or [3, 3]
                - Tau perturbation vector in R^3 ~ se(2) shape [k, 3] or [3].

            output:
            -------
                - M (+) Exp(Tau)
        '''
        exp = self.se2exp(tau)
        #print("\n\t", "-"*5, "SE2 Int", "-"*5)
        #print("\tM:   ", M.shape)
        #print("\texp: ", exp.shape)
        return M @ exp


class SE2Inv(torch.nn.Module):
    def __init__(self):
        super(SE2Inv, self).__init__()
        pass

    def forward(self, M):
        #print("\n\t", "-"*5, "SE2 Inv", "-"*5)
        #print("\tM:    ", M.shape)
        t = M[:, :2, 2:]
        r = M[:, 0:2, 0:2]
        #print("\tt:     ", t.shape)
        #print("\tr:     ", r.shape)
        r_inv = torch.transpose(r, -1, -2)
        #print("\tr_inv: ", r_inv.shape)
        t_inv = -r_inv @ t
        #print("\tt_inv: ", t_inv.shape)
        M_inv = M.clone()
        M_inv[:, :2, 2:] = t_inv
        M_inv[:, 0:2, 0:2] = r_inv
        #print("\tM_inv: ", M_inv.shape)
        return M_inv


class SE2Adj(torch.nn.Module):
    def __init__(self):
        super(SE2Adj, self).__init__()
        skew = torch.Tensor([[[0., -1.], [1., 0]]])
        self.register_buffer("skew", skew)

    def forward(self, M):
        #print("\n\t", "-"*5, "Adjoint", "-"*5)
        t = M[:, :2, 2:]
        #print("\tM:     ", M.shape)
        #print("\tt:     ", t.shape)
        t_adj = - self.skew @ t
        #print("\tt_adj: ", t_adj.shape)
        adj = M.clone()
        adj[:, :2, 2:] = t_adj
        #print("\tadj:   ", adj.shape)
        return adj


class FlattenSE2(torch.nn.Module):
    def __init__(self):
        super(FlattenSE2, self).__init__()
        self.se2log = SE2Log()

    def forward(self, M, v):
        pose = self.se2log(M)
        res = torch.concat([pose, v[:, None]], dim=-1)
        return res


# SE3
class ToSE3Mat(torch.nn.Module):
    def __init__(self):
        super(ToSE3Mat, self).__init__()
        self.se3exp = SE3Exp()

    def forward(self, s):
        '''
            transforms a s \in R^{6} into a Lie Element M \in R^{4*4}
            
            inputs:
            -------
                - s, the state in the R^{k*6} (x, y, z, rot_vec). Shape [k, 6]
            
            outputs:
            --------
                - M, a element of the Lie Group. Shape [k, 4, 4]
        '''
        return self.se3exp(s)


class Skew3(torch.nn.Module):
    def __init__(self):
        super(Skew3, self).__init__()
        # Generators for skew
        e0 = torch.Tensor([[0., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        self.register_buffer('e0', e0)

        e1 = torch.Tensor([[0., 0., 1.], [0., 0., 0.], [-1., 0., 0.]])
        self.register_buffer('e1', e1)

        e2 = torch.Tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 0.]])
        self.register_buffer('e2', e2)
    
    def forward(self, vec):
        '''
            Computes the skew-symetric matrix of the vector vec

            input:
            ------
                - vec. batch of vectors. Shape [k, 3].

            output:
            -------
                - skew(vec) a skew symetric matrix. Shape [k, 3, 3]
        '''

        a = self.e0 * vec[:, 0, None, None]
        b = self.e1 * vec[:, 1, None, None]
        c = self.e2 * vec[:, 2, None, None]

        return a + b + c


class SO3Exp(torch.nn.Module):
    def __init__(self):
        super(SO3Exp, self).__init__()
        self.skew = Skew3()
        a = torch.eye(3)
        self.register_buffer("a", a)
    
    def forward(self, tau):
        '''
            Computes the exponential map of Tau in SO(3).

            input:
            ------
                - tau: perturbation in so(3). shape [k, 3]

            output:
            -------
                - Exp(Tau). shape [k, 3, 3]
        '''

        theta = torch.linalg.norm(tau, dim=1)
        u = normalize(tau, dim=-1)

        skewU = self.skew(u)
        b = torch.sin(theta)[:, None, None]*skewU
        c = (1-torch.cos(theta)[:, None, None])*torch.pow(skewU, 2)

        return self.a + b + c


class SO3Log(torch.nn.Module):
    def __init__(self):
        super(SO3Log, self).__init__()
    
    def forward(self, R):
        trace = R.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
        theta = torch.acos(
            torch.clip(
                ((trace -1.) / 2.),
            -1, 1)
        )
        rot_vec = theta*(R - torch.transpose(R, dim1=-1, dim2=-2)) / (2*torch.sin(theta))


class SE3Exp(torch.nn.Module):
    def __init__(self):
        super(SE3Exp, self).__init__()
    
    def forward(self):
        pass


class SE3Log(torch.nn.Module):
    def __init__(self):
        super(SE3Log, self).__init__()
    
    def forward(self):
        pass


class SE3V(torch.nn.Module):
    def __init__(self):
        super(SE3V, self).__init__()
    
    def forward(self):
        pass


class SE3Int(torch.nn.Module):
    def __init__(self):
        super(SE3Int, self).__init__()
    
    def forward(self):
        pass


class SE3Inv(torch.nn.Module):
    def __init__(self):
        super(SE3Inv, self).__init__()
    
    def forward(self):
        pass


class SE3Adj(torch.nn.Module):
    def __init__(self):
        super(SE3Adj, self).__init__()
    
    def forward(self):
        pass


class FlattenSE3(torch.nn.Module):
    def __init__(self):
        super(FlattenSE3, self).__init__()
    
    def forward(self):
        pass


def plot_traj(traj_dict, plot_cols, tau, fig=False, title="State Evolution", save=False):
    fig_state = plt.figure(figsize=(10, 10))
    axs_states = {}
    for i, name in enumerate(plot_cols):
        m, n = np.unravel_index(i, (2, 3))
        idx = 1*m + 2*n + 1
        axs_states[name] = fig_state.add_subplot(3, 2, idx)
    
    for k in traj_dict:
        t = traj_dict[k]
        for i, name in enumerate(plot_cols):
            axs_states[name].set_ylabel(f'{name}', fontsize=10)
            if k == 'gt':
                if i == 0:
                    axs_states[name].plot(t[:tau+1, i], marker='.', zorder=-10, label=k)
                else:
                    axs_states[name].plot(t[:tau+1, i], marker='.', zorder=-10)
                axs_states[name].set_xlim([0, tau+1])
            
            else:
                if i == 0:
                    axs_states[name].plot(np.arange(0, tau), t[:tau, plot_cols[name]],
                        marker='.', label=k)
                else:
                    axs_states[name].plot(np.arange(0, tau), t[:tau, plot_cols[name]],
                        marker='.')
    fig_state.text(x=0.5, y=0.03, s="steps", fontsize=10)
    fig_state.suptitle(title, fontsize=10)
    fig_state.legend(fontsize=5)
    fig_state.tight_layout(rect=[0, 0.05, 1, 0.98])

    if save:
        fig_state.savefig("img/" + title + ".png")

    if fig:
        return fig_state

    canvas = FigureCanvas(fig_state)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig_state.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return img
