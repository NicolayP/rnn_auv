import torch
from torch.nn.functional import normalize
torch.autograd.set_detect_anomaly(True)

from tqdm import tqdm
import os
import random
import warnings

import numpy as np
import pandas as pd
npdtype = np.float32
tdtype = torch.float32


import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy.spatial.transform import Rotation as R
import yaml

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

# MISC FILES
def read_files(data_dir, files, type="train"):
    dfs = []
    for f in tqdm(files, desc=f"Dir {type}", ncols=150, colour="blue"):
        csv_file = os.path.join(data_dir, f)
        df = pd.read_csv(csv_file)
        df = df.astype(npdtype)
        dfs.append(df)
    return dfs

# TRAJECTORY PLOTTING
def gen_imgs_3D(t_dict, v_dict, dv_dict=None, tau=100):
    '''
    Plots trajectories with euler representation and velocity profiles from 
    the trajectory and velocity dictionaries respectively.
    The trajectories are plotted with length tau
    '''
    plotState={"x(m)":0, "y(m)": 1, "z(m)": 2, "roll(rad)": 3, "pitch(rad)":4, "yaw(rad)": 5}
    plotVels={"u(m/s)":0, "v(m/s)": 1, "w(m/s)": 2, "p(rad/s)": 3, "q(rad/s)": 4, "r(rad/s)": 5}
    plotDVels={"du(m/s)":0, "dv(m/s)": 1, "dw(m/s)": 2, "dp(rad/s)": 3, "dq(rad/s)": 4, "dr(rad/s)": 5}
    t_imgs = []
    v_imgs = []
    if dv_dict is not None:
        dv_imgs = []
    for t in tau:
        t_imgs.append(plot_traj(t_dict, plotState, t))
        v_imgs.append(plot_traj(v_dict, plotVels, t, title="Velcoity Profiles"))
        if dv_dict is not None:
            dv_imgs.append(plot_traj(dv_dict, plotDVels, t, title="Delta V"))

    if dv_dict is not None:
        return t_imgs, v_imgs, dv_imgs

    return t_imgs, v_imgs, dv_imgs


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


def to_euler(traj):
    # assume quaternion representation
    p = traj[..., :3]
    q = traj[..., 3:]
    r = R.from_quat(q)
    e = r.as_euler('xyz')
    return np.concatenate([p, e], axis=-1)

# TRAINING AND VALIDATION
def val_step(dataloader, model, loss, writer, epoch, device):
    torch.autograd.set_detect_anomaly(True)
    size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Val: {epoch}", ncols=200, colour="red", leave=False)
    for batch, data in t:
        X, U, Y = data
        X, U, Y = X.to(device), U.to(device), Y.to(device)
        pred, pred_dv = model(X, U)
        l = loss(pred_dv, Y)

        if writer is not None:
            writer.add_scalar("val-loss/", l, epoch*size+batch)
    
    # Trajectories generation for validation
    gt_trajs, gt_trajs_dv, action_seqs = dataloader.dataset.get_trajs()
    
    x_init = torch.Tensor(gt_trajs[0][None, 0:1, ...]).to(device)
    A = torch.Tensor(action_seqs[0][None, ...]).to(device)

    pred_trajs, pred_trajs_dv = model(x_init, A)

    tau = [10, 20, 30, 40, 50]
    t_dict = {
        "model": to_euler(pred_trajs[0, :, :7].detach().cpu()),
        "gt": to_euler(gt_trajs[0][:, :7])
    }
    v_dict = {
        "model": pred_trajs[0, :, -6:].detach().cpu(),
        "gt": gt_trajs[0][:, -6:]
    }
    dv_dict = {
        "model": pred_trajs_dv[0].detach().cpu(),
        "gt": gt_trajs_dv[0]
    }

    t_imgs, v_imgs, dv_imgs = gen_imgs_3D(t_dict, v_dict, dv_dict, tau=tau)

    dv_losses = [loss(pred_trajs_dv[0, :h], torch.Tensor(gt_trajs_dv[0][:h]).to(device)) for h in tau]
    dv_losses_split = [[loss(pred_trajs_dv[0, :h, dim], torch.Tensor(gt_trajs_dv[0][:h, dim]).to(device)) for dim in range(6)] for h in tau]
    
    # Log Trajs
    for t_img, t in zip(t_imgs, tau):
        writer.add_image(f"traj-{t}", t_img, epoch, dataformats="HWC")
    # Log Vels
    for v_img, t in zip(v_imgs, tau):
        writer.add_image(f"vel-{t}", v_img, epoch, dataformats="HWC")
    # Log dv
    for dv_img, t in zip(dv_imgs, tau):
        writer.add_image(f"dv-{t}", dv_img, epoch, dataformats="HWC")


    name = ["u", "v", "w", "p", "q", "r"]
    for dv_loss, dv_loss_split, t in zip(dv_losses, dv_losses_split, tau):
        writer.add_scalar(f"Multi-step-loss-t{t}/all", dv_loss, epoch)
        for d in range(6):
            writer.add_scalar(f"Multi-step-loss-t{t}/{name[d]}", dv_loss_split[d], epoch)


def train(ds, model, loss_fc, optim, writer, epochs, device, ckpt_dir=None, ckpt_steps=2):
    if writer is not None:
        s = torch.Tensor(np.zeros(shape=(1, 1, 13))).to(device)
        s[..., 6] = 1.
        A = torch.Tensor(np.zeros(shape=(1, 10, 6))).to(device)
        writer.add_graph(model, (s, A))
    size = len(ds[0].dataset)
    l = np.nan
    cur = 0
    t = tqdm(range(epochs), desc="Training", ncols=150, colour="blue",
     postfix={"loss": f"Loss: {l:>7f} [{cur:>5d}/{size:>5d}]"})
    for e in t:
        l, cur = train_step(ds[0], model, loss_fc, optim, writer, e, device)
        if (e % ckpt_steps == 0) and ckpt_dir is not None:
            val_step(ds[1], model, loss_fc, writer, e, device)
            tmp_path = os.path.join(ckpt_dir, f"step_{e}.pth")
            torch.save(model.state_dict(), tmp_path)
        t.set_postfix({"loss": f"Loss: {l:>7f} [{cur:>5d}/{size:>5d}]"})

        if writer is not None:
            writer.flush()


def train_step(dataloader, model, loss, optim, writer, epoch, device):
    #print("\n", "="*5, "Training", "="*5)
    torch.autograd.set_detect_anomaly(True)
    size = len(dataloader.dataset)
    model.train()
    t = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch}", ncols=200, colour="red", leave=False)
    for batch, data in t:
        X, U, Y = data
        X, U, Y = X.to(device), U.to(device), Y.to(device)
        
        pred, pred_dv = model(X, U)
        optim.zero_grad()
        l = loss(pred_dv, Y)
        l.backward()
        optim.step()

        if writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram("train/" + name, param, epoch*size+batch)
                for dim in range(3):
                    loss_dim = loss(pred_dv[..., dim], Y[..., dim])
                    writer.add_scalar("dv-split-loss/" + str(dim), loss_dim, epoch*size+batch)
                writer.add_scalar("train-loss/", l, epoch*size+batch)

    return l.item(), batch*len(X)

# DATASET FOR 3D DATA
class DatasetList3D(torch.utils.data.Dataset):
    def __init__(self, data_list, steps=1, v_frame="body", dv_frame="body", rot="quat", act_normed=False, traj=False):
        super(DatasetList3D, self).__init__()
        self.data_list = data_list
        self.s = steps
        if v_frame == "body":
            v_prefix = "B"
        elif v_frame == "world":
            v_prefix = "I"

        if dv_frame == "body":
            dv_prefix = "B"
        elif dv_frame == "world":
            dv_prefix = "I"

        self.pos = ['x', 'y', "z"]
        # used for our SE3 implementation.
        if rot == "rot":
            self.rot = ['r00', 'r01', 'r02',
                        'r10', 'r11', 'r12',
                        'r20', 'r21', 'r22']
        # Used in pypose implementation.
        elif rot == "quat":
            self.rot = ['qx', 'qy', 'qz', 'qw']

        self.lin_vel = [f'{v_prefix}u', f'{v_prefix}v', f'{v_prefix}w']
        self.ang_vel = [f'{v_prefix}p', f'{v_prefix}q', f'{v_prefix}r']

        self.x_labels = self.pos + self.rot + self.lin_vel + self.ang_vel

        self.y_labels = [
            f'{dv_prefix}du', f'{dv_prefix}dv', f'{dv_prefix}dw',
            f'{dv_prefix}dp', f'{dv_prefix}dq', f'{dv_prefix}dr'
        ]

        if act_normed:
            self.u_labels = ['Ux', 'Uy', 'Uz', 'Vx', 'Vy', 'Vz']
        else:
            self.u_labels = ['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']

        self.samples = [traj.shape[0] - self.s for traj in data_list]
        self.len = sum(self.samples)
        self.bins = self.create_bins()
        self.traj_ret = traj

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        i = (np.digitize([idx], self.bins)-1)[0]
        traj = self.data_list[i]
        j = idx - self.bins[i]
        sub_frame = traj.iloc[j:j+self.s+1]
        x = sub_frame[self.x_labels].to_numpy()
        x = x[:1]

        u = sub_frame[self.u_labels].to_numpy()
        u = u[:self.s]

        y = sub_frame[self.y_labels].to_numpy()
        y = y[1:1+self.s]
        if not self.traj_ret:
            return x, u, y

        traj = sub_frame[self.x_labels].to_numpy()
        traj = traj[1:1+self.s]
        return x, u, y, traj

    @property
    def nb_trajs(self):
        return len(self.data_list)
    
    def get_traj(self, idx):
        if idx >= self.nb_trajs:
            raise IndexError
        return self.data_list[idx][self.x_labels].to_numpy()
    
    def create_bins(self):
        bins = [0]
        cummul = 0
        for s in self.samples:
            cummul += s
            bins.append(cummul)
        return bins

    def get_trajs(self):
        traj_list = []
        dv_traj_list = []
        action_seq_list = []
        for data in self.data_list:
            traj = data[self.x_labels].to_numpy()
            traj_list.append(traj)

            dv_traj = data[self.y_labels].to_numpy()
            dv_traj_list.append(dv_traj)

            action_seq = data[self.u_labels].to_numpy()
            action_seq_list.append(action_seq)
        return traj_list, dv_traj_list, action_seq_list


def parse_param(file):
    with open(file) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf

def save_param(path, params):
    with open(path, "w") as stream:
        yaml.dump(params, stream)