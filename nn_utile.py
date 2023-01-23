import torch
torch.autograd.set_detect_anomaly(True)
import pypose as pp
import numpy as np
import pandas as pd
from utile import tdtype, npdtype
from torch.utils.tensorboard import SummaryWriter


class AUVRNNDeltaVProxy(torch.nn.Module):
    '''
        Proxy for the RNN part of the network. Used to ensure that
        the integration using PyPose is correct.
    '''
    def __init__(self, dv):
        super(AUVRNNDeltaVProxy, self).__init__()
        self._dv = dv
        self.i = 0

    def forward(self, x, v, a, h0=None):
        self.i += 1
        return self._dv[:, self.i-1:self.i], None


class AUVRNNDeltaV(torch.nn.Module):
    '''
        RNN predictor for $\delta v$.

        parameters:
        -----------
            - rnn_layer:
            - rnn_hidden_size:
            - bias: Whether or not to apply bias to the FC units. (default False)
    '''
    def __init__(self, params=None):
        super(AUVRNNDeltaV, self).__init__()

        self.input_size = 9 + 6 + 6 # rotation matrix + velocities + action. I.E 21
        self.output_size = 6

        self.rnn_layers = 5
        self.rnn_hidden_size = 1
        rnn_bias = False
        nonlinearity = "tanh"
        topology = [32, 32]
        fc_bias = False
        bn = True

        if params is not None:
            self.rnn_layers = params["rnn"]["rnn_layer"]
            self.rnn_hidden_size = params["rnn"]["rnn_hidden_size"]
            rnn_bias = params["rnn"]["bias"]
            nonlinearity = params["rnn"]["activation"]
            topology = params["fc"]["topology"]
            fc_bias = params["fc"]["bias"]
            bn = params["fc"]["batch_norm"]

        self.rnn = torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=self.rnn_layers,
            batch_first=True,
            bias=rnn_bias,
            nonlinearity=nonlinearity
        )

        fc_layers = []
        for i, s in enumerate(topology):
            if i == 0:
                layer = torch.nn.Linear(self.rnn_hidden_size, s, bias=fc_bias)
            else:
                layer = torch.nn.Linear(topology[i-1], s, bias=fc_bias)
            fc_layers.append(layer)

            # TODO try batch norm.
            if bn:
                fc_layers.append(torch.nn.BatchNorm1d(s))

            fc_layers.append(torch.nn.LeakyReLU(negative_slope=0.1))

        layer = torch.nn.Linear(topology[-1], 6, bias=fc_bias)
        fc_layers.append(layer)

        self.fc = torch.nn.Sequential(*fc_layers)
        #self.fc.apply(init_weights)

    def forward(self, x, v, a, h0=None):
        # print("\t\t", "="*5, "DELTA V", "="*5)

        k = x.shape[0]
        r = x.rotation().matrix().flatten(start_dim=-2)
        input_seq = torch.concat([r, v, a], dim=-1)

        if h0 is None:
            h0 = self.init_hidden(k, x.device)

        out, hN = self.rnn(input_seq, h0)
        dv = self.fc(out[:, 0])
        return dv[:, None], hN

    def init_hidden(self, k, device):
        return torch.zeros(self.rnn_layers, k, self.rnn_hidden_size).to(device)


def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        #m.bias.data.fill_(0.01)
    pass


class AUVStep(torch.nn.Module):
    def __init__(self, params=None, dt=0.1, v_frame=None, dv_frame=None):
        super(AUVStep, self).__init__()
        if params is not None:
            self.dv_pred = AUVRNNDeltaV(params["model"])
            self.v_frame = params["dataset_params"]["v_frame"]
            self.dv_frame = params["dataset_params"]["dv_frame"]
        else:
            self.dv_pred = AUVRNNDeltaV()
            self.v_frame = v_frame
            self.dv_frame = dv_frame
        self.dt = dt
        self.std = 1.
        self.mean = 0.

    def forward(self, x, v, a, h0=None):
        '''
            Predicts next state (x_next, v_next) from (x, v, a, h0)

            inputs:
            -------
                - x, pypose.SE3. The current pose on the SE3 manifold.
                    shape [k, 7] (pypose uses quaternion representation)
                - v, torch.Tensor \in \mathbb{R}^{6} \simeq pypose.se3.
                    The current velocity. shape [k, 6]
                - a, torch.Tensor the current applied forces.
                    shape [k, 6]
                - h0, the hidden state of the RNN network.
            
            outputs:
            --------
                - x_next, pypose.SE3. The next pose on the SE3 manifold.
                    shape [k, 7]
                - v_next, torch.Tensor \in \mathbb{R}^{6} \simeq pypose.se3.
                    The next velocity. Shape [k, 6]
                - h_next, 
        '''
        dv, h_next = self.dv_pred(x, v, a, h0)

        dv_unnormed = dv*self.std + self.mean

        t = pp.se3(self.dt*v).Exp()
        if self.v_frame == "body":
            x_next = x * t
        elif self.v_frame == "world":
            x_next = t * x

        if self.v_frame == self.dv_frame:
            v_next = v + dv_unnormed
        elif self.v_frame == "world": # assumes that dv is in body frame.
            v_next = v + x.Adj(dv_unnormed)
        elif self.v_frame == "body": # assumes that dv is in world frame.
            v_next = v + x.Inv().Adj(dv_unnormed)

        return x_next, v_next, dv, h_next                     

    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std


class AUVTraj(torch.nn.Module):
    def __init__(self, params=None, se3=True):
        super(AUVTraj, self).__init__()
        self.step = AUVStep(params)
        if params is not None:
            self.se3 = params["model"]["se3"]
        else:
            self.se3 = True

    def forward(self, s, A):
        '''
            Generates a trajectory using a Inital State (pose + velocity) combined
            with an action sequence.

            inputs:
            -------
                - p, torch.Tensor. The pose of the system with quaternion representation.
                    shape [k, 7]
                - v, torch.Tensor. The velocity (in body frame) of the system.
                    shape [k, 6]
                - A, torch.Tensor. The action sequence appliedto the system.
                    shape [k, Tau, 6]

            outputs:
            --------
                - traj, torch.Tensor. The generated trajectory.
                    shape [k, tau, 7]
                - traj_v, torch.Tensor. The generated velocity profiles.
                    shape [k, tau, 6]
                - traj_dv, torch.Tensor. The predicted velocity delta. Used for
                    training. shape [k, tau, 6]
        '''
        k = A.shape[0]
        tau = A.shape[1]
        h = None
        p = s[..., :7]
        v = s[..., 7:]
        traj = torch.zeros(size=(k, tau, 7)).to(p.device)
        if self.se3:
            traj = pp.SE3(traj)
        traj_v = torch.zeros(size=(k, tau, 6)).to(p.device)
        traj_dv = torch.zeros(size=(k, tau, 6)).to(p.device)
        
        x = pp.SE3(p).to(p.device)
        for i in range(tau):
            # print("="*5, f"Step {i}", "="*5)
            #print("x:      ", x.shape)
            #print("v:      ", v.shape)
            #print("A:      ", A[:, i:i+1].shape)
            # i:i+1 is to keep the dimension to match other inputs
            x_next, v_next, dv, h_next = self.step(x, v, A[:, i:i+1], h)

            # print("x_next: ", x.shape)
            # print("v_next: ", v.shape)
            # print("dv:     ", dv.shape)
            #print("h_next: ", h_next.shape)

            x, v, h = x_next, v_next, h_next

            if self.se3:
                traj[:, i:i+1] = x
            else:
                traj[:, i:i+1] = x.data

            traj_v[:, i:i+1] = v
            traj_dv[:, i:i+1] = dv
        return traj, traj_v, traj_dv


class GeodesicLoss(torch.nn.Module):
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    def forward(self, X1, X2):
        d = (X1 * X2.Inv()).Log()
        square = torch.pow(d, 2)
        return square


class TrajLoss(torch.nn.Module):
    def __init__(self, alpha=1., beta=0., gamma=0.):
        super(TrajLoss, self).__init__()
        self.l2 = torch.nn.MSELoss()
        self.geodesic = GeodesicLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        pass

    def has_v(self):
        return self.beta > 0.

    def has_dv(self):
        return self.gamma > 0.

    def forward(self, traj1, traj2, v1=None, v2=None, dv1=None, dv2=None, split=False):
        '''
            Computes loss on an entire trajectory. Optionally if
            dv is passed, it computes the loss on the velocity delta.

            inputs:
            -------
                traj1: pypose SE(3) elements sequence representing first trajectory
                    shape [k, tau]
                traj2: pypose SE(3) elements sequence representing second trajectory
                    shape [k, tau]
                v1: pytorch Tensor. velocity profiles
                    shape [k, tau, 6]
                v2: pytorch Tensor. velocity profiles
                    shape [k, tau, 6]
                dv1: pytorch Tensor. Delta velocities profiles
                    shape [k, tau, 6]
                dv2: pytorch Tensor. Delta velocities profiles
                    shape [k, tau, 6]
                split: bool (default = False), if true, returns the loss function
                    splitted across each controlled dimension

        '''
        if split:
            return self.split_loss(traj1, traj2, v1, v2, dv1, dv2)
        return self.loss(traj1, traj2, v1, v2, dv1, dv2)

    def split_loss(self, t1, t2, v1, v2, dv1, dv2):
        # only used for logging and evaluating the performances.
        t_l = self.geodesic(t1, t2).mean((0, 1))
        v_l = torch.pow(v1 - v2, 2).mean((0, 1))
        dv_l = torch.pow(dv1 - dv2, 2).mean((0, 1))
        return t_l, v_l, dv_l

    def loss(self, t1, t2, v1, v2, dv1, dv2):
        t_l = self.geodesic(t1, t2).mean()
        v_l = self.l2(v1, v2).mean()
        dv_l = self.l2(dv1, dv2).mean()
        return self.alpha*t_l + self.beta*v_l + self.gamma*dv_l


# DATASET FOR 3D DATA
class DatasetList3D(torch.utils.data.Dataset):
    def __init__(self, data_list, steps=1,
                 v_frame="body", dv_frame="body", rot="quat",
                 act_normed=False, se3=False, out_normed=True, stats=None):
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

        self.traj_labels = self.pos + self.rot
        self.vel_labels = self.lin_vel + self.ang_vel
        self.dv_labels = [
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
        self.se3 = se3
        
        if out_normed:
            self.std = np.array(stats["std"][f'{dv_prefix}_norm'], dtype=npdtype)
            self.mean = np.array(stats["mean"][f'{dv_prefix}_norm'], dtype=npdtype)
        else:
            self.std = 1.
            self.mean = 0.

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

        traj = sub_frame[self.traj_labels].to_numpy()[1:1+self.s]
        vel = sub_frame[self.vel_labels].to_numpy()[1:1+self.s]
        dv = sub_frame[self.dv_labels].to_numpy()[1:1+self.s]

        dv = (dv-self.mean)/self.std

        if self.se3:
            traj = pp.SE3(traj)

        return x, u, traj, vel, dv

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
        vel_list = []
        dv_list = []
        action_seq_list = []
        for data in self.data_list:
            traj = data[self.traj_labels].to_numpy()[None]
            traj_list.append(traj)

            vel = data[self.vel_labels].to_numpy()[None]
            vel_list.append(vel)

            dv = data[self.dv_labels].to_numpy()[None]
            dv_list.append(dv)

            action_seq = data[self.u_labels].to_numpy()[None]
            action_seq_list.append(action_seq)

        trajs = torch.Tensor(np.concatenate(traj_list, axis=0))
        vels = torch.Tensor(np.concatenate(vel_list, axis=0))
        dvs = torch.Tensor(np.concatenate(dv_list, axis=0))
        actions = torch.Tensor(np.concatenate(action_seq_list, axis=0))

        dvs = (dvs-self.mean)/self.std

        if self.se3:
            trajs = pp.SE3(trajs)

        return trajs, vels, dvs, actions

    def get_stats(self):
        return self.mean, self.std