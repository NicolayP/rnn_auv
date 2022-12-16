import torch
torch.autograd.set_detect_anomaly(True)
import pypose as pp
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from utile import read_files, DatasetList3D, train

from tqdm import tqdm
import os
import random
random.seed(0)
import warnings

npdtype = np.float32
tdtype = torch.float32

from utile import plot_traj
import matplotlib.pyplot as plt

import time


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
        return self._dv[:, self.i:self.i+1], None


class AUVRNNDeltaV(torch.nn.Module):
    '''
        RNN predictor for $\delta v$.

        parameters:
        -----------
            - rnn_layer:
            - rnn_hidden_size:
            - bias: Whether or not to apply bias to the FC units. (default False)
    '''
    def __init__(self, bias=False, rnn_hidden_size=5, rnn_layers=1):
        super(AUVRNNDeltaV, self).__init__()

        self.input_size = 9 + 6 + 6 # rotation matrix + velocities + action
        self.output_size = 6
        self.rnn_layers = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.rnn = torch.nn.RNN(
            input_size=self.input_size,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_layers,
            batch_first=True,
            bias=bias
        )

        fc_layers = [
            torch.nn.Linear(rnn_hidden_size, 32, bias=bias),
            torch.nn.LeakyReLU(negative_slope=0.1),
            torch.nn.Linear(32, self.output_size, bias=bias)
        ]
        self.fc = torch.nn.Sequential(*fc_layers)
    
    def forward(self, x, v, a, h0=None):
        # print("\t\t", "="*5, "DELTA V", "="*5)

        k = x.shape[0]
        r = x.rotation().matrix().flatten(start_dim=-2)
        input_seq = torch.concat([r, v, a], dim=-1)

        # print("\t\tx:         ", x.shape)
        # print("\t\tr:         ", r.shape)
        # print("\t\tv:         ", v.shape)
        # print("\t\ta:         ", a.shape)

        if h0 is None:
            h0 = self.init_hidden(k, x.device)

        
        # print("\t\tinput_seq: ", input_seq.shape)
        # print("\t\th0:        ", h0.shape)

        out, hN = self.rnn(input_seq, h0)
        # print("\t\tout:       ", out.shape)
        # print("\t\thN:        ", hN.shape)
        dv = self.fc(out[:, 0])
        # print("\t\tdv:        ", dv.shape)

        return dv[:, None], hN

    def init_hidden(self, k, device):
        return torch.zeros(self.rnn_layers, k, self.rnn_hidden_size).to(device)


class AUVStep(torch.nn.Module):
    def __init__(self, dt=0.1, dv_frame="body"):
        super(AUVStep, self).__init__()
        self.dv_pred = AUVRNNDeltaV()
        self.dt = dt
        self.dv_frame = dv_frame

        if self.dv_frame == "body":
            self.forward = self.forward_body
        elif self.dv_frame == "world":
            self.forward = self.forward_inertial

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
        pass                                                                

    def forward_body(self, x, v, a, h0=None):
        # print("\t", "="*5, "AUV STEP", "="*5)
        # print("\tx:      ", x.shape)
        # print("\tv:      ", v.shape)
        # print("\tA:      ", a.shape)
        
        #print("h:      ", h0.shape)
        Bdv, h_next = self.dv_pred(x, v, a, h0)

        #print("\tBdv:    ", Bdv.shape)
        #print("\th_next: ", h_next.shape)
        # compute displacement.
        t = v*self.dt

        #print("\tt:      ", t.shape)
        # Update pose using right \oplus operator as we assume
        # the velocity to be in body frame.
        x_next = x * pp.se3(t).Exp()

        #print("\tx_next: ", x_next.shape)
        # Update the velocity.0
        v_next = v + Bdv

        # print("\tv_next: ", v_next.shape)
        return x_next, v_next, Bdv, h_next

    def forward_inertial(self, x, v, a, h0=None):
        Idv, h_next = self.dv_pred(x, v, a, h0)
        # Compute the displacement.
        t = v*self.dt
        # Update pose using right \oplus operator as we assume the 
        # velocity to be in body frame.
        x_next = x * pp.se3(t).Exp()
        # Update the velocity vector.
        # Bring the current velocity in the lie algebra
        I_v = x.Adj(v)
        # Add the velocity delta (assumed in world frame)
        I_v_next = I_v + Idv
        # Revert the new velocity back into the body frame.
        v_next = x.Inv().Adj(I_v_next)
        return x_next, v_next, Idv, h_next


class AUVTraj(torch.nn.Module):
    def __init__(self):
        super(AUVTraj, self).__init__()
        self.step = AUVStep(dv_frame="body")

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
        traj = torch.zeros(size=(k, tau, 7+6)).to(p.device)
        traj_dv = torch.zeros(size=(k, tau, 6)).to(p.device)
        
        x = pp.SE3(p).to(p.device)

        for i in range(tau):
            #print("="*5, "Step", "="*5)
            #print("x:      ", x.shape)
            #print("v:      ", v.shape)
            #print("A:      ", A[:, i:i+1].shape)
            #print("h:      ", h.shape)
            # i:i+1 is to keep the dimension to match other inputs
            x_next, v_next, dv, h_next = self.step(x, v, A[:, i:i+1], h)

            #print("x_next: ", x.shape)
            #print("v_next: ", v.shape)
            #print("dv:     ", dv.shape)
            #print("h_next: ", h_next.shape)

            x, v, h = x_next, v_next, h_next
            traj[:, i:i+1] = torch.concat([x.data, v], axis=-1)
            traj_dv[:, i:i+1] = dv
        return traj, traj_dv


def get_device(gpu=False):
    use_cuda = False
    if gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")
    return torch.device("cuda:0" if use_cuda else "cpu")


def integrate():
    tau = 300
    device = get_device(gpu=True)
    dir = 'test_data_clean/csv/'
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    trajs = []
    trajs_plot = []
    vels = []
    acts = []
    Idvs = []
    Bdvs = []

    for f in files:
        df = pd.read_csv(f)
        trajs.append(torch.Tensor(df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].to_numpy())[None])
        trajs_plot.append(df[['x', 'y', 'z', 'roll', 'pitch', 'yaw']].to_numpy()[None])
        vels.append(torch.Tensor(df[['Bu', 'Bv', 'Bw', 'Bp', 'Bq', 'Br']].to_numpy())[None])
        acts.append(torch.Tensor(df[['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']].to_numpy())[None])
        Idvs.append(torch.Tensor(df[['Idu', 'Idv', 'Idw', 'Idp', 'Idq', 'Idr']].to_numpy())[None])
        Bdvs.append(torch.Tensor(df[['Bdu', 'Bdv', 'Bdw', 'Bdp', 'Bdq', 'Bdr']].to_numpy())[None])

    trajs = torch.concat(trajs, dim=0).to(device)
    trajs_plot = np.concatenate(trajs_plot, axis=-1)
    vels = torch.concat(vels, dim=0).to(device)
    acts = torch.concat(acts, dim=0).to(device)
    Idvs = torch.concat(Idvs, dim=0).to(device)
    Bdvs = torch.concat(Bdvs, dim=0).to(device)

    #dv_pred = AUVRNNDeltaVProxy(Bdvs).to(device)
    #dv_pred = AUVRNNDeltaV().to(device)

    #step = AUVStep(dv_frame="body").to(device)
    #step.dv_pred = dv_pred

    pred = AUVTraj().to(device)
    #pred.step = step

    input = torch.concat([trajs[:, :1], vels[:, :1]], dim=-1)

    print("input: ", input.shape)
    print("acts:  ", acts[:, :2].shape)
    pred(input, acts[:, :-1])

    log_path = "train_log/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    writer.add_graph(pred, (input, acts[:, :2]))

    print(input.shape)
    print(acts[:, :-1].shape)

    s = time.time()
    pred_trajs, pred_dvs = pred(input, acts[:, :-1])
    e = time.time()
    print(f"Prediction time: {e-s}")

    pred_trajs, pred_dvs = pred_trajs.detach().cpu(), pred_dvs.detach().cpu()
    trajs, vels, Idvs, Bdvs = trajs.cpu(), vels.cpu(), Idvs.cpu(), Bdvs.cpu()

    pred_vs = pred_trajs[..., -6:]
    pred_trajs = pred_trajs[..., :-6]

    # plotting
    pred_traj_euler = to_euler(pred_trajs[0])
    traj_euler = trajs_plot[0]
    s_col = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
    plot_traj({"pred": pred_traj_euler, "gt": traj_euler}, s_col, tau, True)
    v_col = {"u": 0, "v": 1, "w": 2, "p": 3, "q": 4, "r": 5}
    plot_traj({"pred": pred_vs[0], "gt": vels[0]}, v_col, tau, True)
    dv_col = {"Bdu": 0, "Bdv": 1, "Bdw": 2, "Bdp": 3, "Bdq": 4, "Bdr": 5}
    plot_traj({"pred": pred_dvs[0], "gt": Bdvs[0]}, dv_col, tau, True)
    plt.show()
    

    return


def training():
    nb_files = 100
    data_dir = 'csv'
    dir_name = os.path.basename(data_dir)
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    random.shuffle(files)
    files = random.choice(files, k=nb_files)

    # split train and val in 70-30 ration
    train_size = int(0.7*len(files))
    
    train_files = files[:train_size]
    val_files = files[train_size:]

    print("Data size:  ", len(files))
    print("Train size: ", len(train_files))
    print("Val size:   ", len(val_files))

    dfs_train = read_files(data_dir, train_files, "train")
    dataset_train = DatasetList3D(dfs_train, steps=10)

    dfs_val = read_files(data_dir, val_files, "val")
    dataset_val = DatasetList3D(dfs_val, steps=10)

    train_params = {"batch_size": 2048, "shuffle": True, "num_workers": 8}

    ds = (
        torch.utils.data.DataLoader(dataset_train, **train_params),
        torch.utils.data.DataLoader(dataset_val, **train_params)
    )

    log_path = "train_log/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    ckpt_dir = "train_log/train_ckpt/"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    device = get_device(True)
    model = AUVTraj().to(device)
    loss_fc = torch.nn.MSELoss().to(device)
    optim = torch.optim.Adam(model.parameters())
    epochs = 1

    train(ds, model, loss_fc, optim, writer, epochs, device, ckpt_dir)


if __name__ == "__main__":
    training()
    