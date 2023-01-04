import torch
torch.autograd.set_detect_anomaly(True)
import pypose as pp
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from utile import read_files, DatasetList3D, train, parse_param, save_param, to_euler

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
from datetime import datetime

import argparse


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

        if params is not None:
            self.rnn_layers = params["rnn"]["rnn_layer"]
            self.rnn_hidden_size = params["rnn"]["rnn_hidden_size"]
            rnn_bias = params["rnn"]["bias"]
            nonlinearity = params["rnn"]["activation"]
            topology = params["fc"]["topology"]
            fc_bias = params["fc"]["bias"]

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
            if params["fc"]["batch_norm"]:
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

        t = pp.se3(v*self.dt).Exp()
        if self.v_frame == "body":
            x_next = x * t
        elif self.v_frame == "world":
            x_next = t * x

        if self.v_frame == self.dv_frame:
            v_next = v + dv
        elif self.v_frame == "world": # assumes that dv is in body frame.
            v_next = v + x.Adj(dv)
        elif self.v_frame == "body": # assumes that dv is in world frame.
            v_next = v + x.Inv().Adj(dv)

        return x_next, v_next, dv, h_next                             


class AUVTraj(torch.nn.Module):
    def __init__(self, params=None):
        super(AUVTraj, self).__init__()
        self.step = AUVStep(params)

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
    tau = 500
    device = get_device(gpu=True)
    dir = 'test_data_clean/2csv/'
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    trajs = []
    trajs_plot = []
    Ivels = []
    Bvels = []
    acts = []
    Idvs = []
    Bdvs = []

    for f in files:
        df = pd.read_csv(f)
        trajs.append(torch.Tensor(df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].to_numpy())[None])
        trajs_plot.append(df[['x', 'y', 'z', 'roll', 'pitch', 'yaw']].to_numpy()[None])
        Ivels.append(torch.Tensor(df[['Iu', 'Iv', 'Iw', 'Ip', 'Iq', 'Ir']].to_numpy())[None])
        Bvels.append(torch.Tensor(df[['Bu', 'Bv', 'Bw', 'Bp', 'Bq', 'Br']].to_numpy())[None])
        acts.append(torch.Tensor(df[['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']].to_numpy())[None])
        Idvs.append(torch.Tensor(df[['Idu', 'Idv', 'Idw', 'Idp', 'Idq', 'Idr']].to_numpy())[None])
        Bdvs.append(torch.Tensor(df[['Bdu', 'Bdv', 'Bdw', 'Bdp', 'Bdq', 'Bdr']].to_numpy())[None])

    trajs = torch.concat(trajs, dim=0).to(device)
    trajs_plot = np.concatenate(trajs_plot, axis=-1)
    Ivels = torch.concat(Ivels, dim=0).to(device)
    Bvels = torch.concat(Bvels, dim=0).to(device)
    acts = torch.concat(acts, dim=0).to(device)
    Idvs = torch.concat(Idvs, dim=0).to(device)
    Bdvs = torch.concat(Bdvs, dim=0).to(device)

    dv_pred = AUVRNNDeltaVProxy(Bdvs).to(device)
    #dv_pred = AUVRNNDeltaV().to(device)

    step = AUVStep(v_frame="body", dv_frame="body").to(device)
    step.dv_pred = dv_pred

    pred = AUVTraj().to(device)
    pred.step = step

    input = torch.concat([trajs[:, :1], Bvels[:, :1]], dim=-1)

    print("input: ", input.shape)
    print("acts:  ", acts.shape)
    pred.step.dv_pred.i = 0
    foo, bar = pred(input, acts)

    #exit()

    #log_path = "integrate_log/"
    #if not os.path.exists(log_path):
    #    os.makedirs(log_path)
    #writer = SummaryWriter(log_path)

    #writer.add_graph(pred, (input, acts[:, :2]))

    pred.step.dv_pred.i = 0
    s = time.time()
    pred_trajs, pred_dvs = pred(input, acts)
    e = time.time()
    print(f"Prediction time: {e-s}")

    pred_trajs, pred_dvs = pred_trajs.detach().cpu(), pred_dvs.detach().cpu()
    trajs, Ivels, Bvels, Idvs, Bdvs = trajs.cpu(), Ivels.cpu(), Bvels.cpu(), Idvs.cpu(), Bdvs.cpu()

    pred_vs = pred_trajs[..., -6:]
    pred_trajs = pred_trajs[..., :-6]

    # plotting
    pred_traj_euler = to_euler(pred_trajs[0])
    traj_euler = trajs_plot[0]

    s_col = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
    plot_traj({"pred": pred_traj_euler, "gt": traj_euler}, s_col, tau, True, title="State")
    v_col = {"u": 0, "v": 1, "w": 2, "p": 3, "q": 4, "r": 5}
    plot_traj({"pred": pred_vs[0], "gt": Bvels[0]}, v_col, tau, True, title="Velocities")
    dv_col = {"Bdu": 0, "Bdv": 1, "Bdw": 2, "Bdp": 3, "Bdq": 4, "Bdr": 5}
    plot_traj({"pred": pred_dvs[0], "gt": Bdvs[0]}, dv_col, tau, True, title="Velocities Deltas")
    plt.show()

    return


def training(params):
    nb_files = params["dataset_params"]["samples"]
    data_dir = params["dataset_params"]["dir"]

    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    random.shuffle(files)

    nb_files = min(len(files), nb_files)

    files = random.sample(files, nb_files)

    # split train and val in 70-30 ration
    train_size = int(params["dataset_params"]["split"]*len(files))

    train_files = files[:train_size]
    val_files = files[train_size:]

    print("Data size:  ", len(files))
    print("Train size: ", len(train_files))
    print("Val size:   ", len(val_files))


    dfs_train = read_files(data_dir, train_files, "train")
    dataset_train = DatasetList3D(
        dfs_train,
        steps=params["dataset_params"]["steps"],
        v_frame=params["dataset_params"]["v_frame"],
        dv_frame=params["dataset_params"]["dv_frame"],
        act_normed=params["dataset_params"]["act_normed"]
    )

    dfs_val = read_files(data_dir, val_files, "val")
    dataset_val = DatasetList3D(
        dfs_val,
        steps=params["dataset_params"]["steps"],
        v_frame=params["dataset_params"]["v_frame"],
        dv_frame=params["dataset_params"]["dv_frame"],
        act_normed=params["dataset_params"]["act_normed"]
    )

    train_params = params["data_loader_params"]

    ds = (
        torch.utils.data.DataLoader(dataset_train, **train_params),
        torch.utils.data.DataLoader(dataset_val, **train_params)
    )

    log_path = params["log"]["path"]
    if params["log"]["stamped"]:
        stamp = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        log_path = os.path.join(log_path, stamp)
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    # TODO: save parameter file in log_path
    save_param(os.path.join(log_path, "parameters.yaml"), params)

    writer = SummaryWriter(log_path)

    ckpt_dir = os.path.join(log_path, "train_ckpt")
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    device = get_device(True)
    model = AUVTraj(params).to(device)
    loss_fc = torch.nn.MSELoss().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=params["optim"]["lr"])
    epochs = params["optim"]["epochs"]

    train(ds, model, loss_fc, optim, writer, epochs, device, ckpt_dir)


def verify_ds():
    # Create dataset of 10-15 steps with one or two trajs.
    tau = 10
    v_frame = "body"
    dv_frame = "body"
    data_dir = "test_data_clean/2csv"
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    dfs_verif = read_files(data_dir, files, "verif")
    verif_ds = DatasetList3D(
        dfs_verif,
        steps=tau,
        v_frame=v_frame,
        dv_frame=dv_frame,
        traj=True
    )


    step = AUVStep(v_frame=v_frame, dv_frame=dv_frame)
    pred = AUVTraj()
    pred.step = step

    loss = torch.nn.MSELoss()
    error = torch.zeros(size=(1, tau, 13))
    # Create a DV predictor for every entry.

    print(len(verif_ds))
    for data in verif_ds:
        x, u, y, traj = data
        x = torch.tensor(x)[None]
        u = torch.tensor(u)[None]
        y = torch.tensor(y)[None]
        traj = torch.tensor(traj)[None]

        # print("X: ", x.shape)
        # print("U: ", u.shape)
        # print("Y: ", y.shape)
        # print("traj: ", traj.shape)


        dv_pred = AUVRNNDeltaVProxy(y)
        pred.step.dv_pred = dv_pred
        pred.step.dv_pred.i = 0
        # Integrate the DV predictor
        pred_trajs, pred_dvs = pred(x, u)

        #print("pred_traj", pred_trajs.shape)

        error += loss(traj, pred_trajs)


        pred_trajs, pred_dvs = pred_trajs.detach().cpu(), pred_dvs.detach().cpu()
        traj = traj.detach().cpu()

        pred_vs = pred_trajs[..., -6:]
        pred_trajs = pred_trajs[..., :-6]

        Bvels = traj[..., -6:]
        traj = traj[..., :-6]
        Bdvs = y

        pred_traj_euler = to_euler(pred_trajs[0])
        traj_euler = to_euler(traj[0])

        # Can investigate every sup prediciton to be sure.
        # s_col = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
        # plot_traj({"pred": pred_traj_euler, "gt": traj_euler}, s_col, tau, True, title="State")
        # v_col = {"u": 0, "v": 1, "w": 2, "p": 3, "q": 4, "r": 5}
        # plot_traj({"pred": pred_vs[0], "gt": Bvels[0]}, v_col, tau, True, title="Velocities")
        # dv_col = {"Bdu": 0, "Bdv": 1, "Bdw": 2, "Bdp": 3, "Bdq": 4, "Bdr": 5}
        # plot_traj({"pred": pred_dvs[0], "gt": Bdvs[0]}, dv_col, tau, True, title="Velocities Deltas")
        # plt.show()


    # Compare it to the Expected outcome.

    #print(error/len(verif_ds))

    error_vs = error[..., -6:]
    error_trajs = error[..., :-6]

    #not sure about that one
    error_euler = to_euler(error_trajs[0])
    s_col = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
    plot_traj({"pred": error_euler}, s_col, tau, True, title="State")
    v_col = {"u": 0, "v": 1, "w": 2, "p": 3, "q": 4, "r": 5}
    plot_traj({"pred": error_vs[0]}, v_col, tau, True, title="Velocities")
    plt.show()

    pass


def parse_arg():
    parser = argparse.ArgumentParser(prog="RNN-AUV",
                                     description="Trains AUV in 3D using pypose.")

    parser.add_argument('-i', '--integrate', action=argparse.BooleanOptionalAction,
                        help="If set, this will run integration function on the dataset.\
                        It helps to see if everything is working as expected")

    parser.add_argument('-d', '--datadir', type=str, default=None,
                        help="dir containing the csv files of the trajectories.")

    parser.add_argument('-l', '--log', type=str, default="train_log",
                        help="directory for the log. Contains tensorboard log, \
                        hyper parameters in yaml file. And training checkpoints.")

    parser.add_argument("-f", "--frame", type=str,
                        help="The frame in which the data will be loaded default body",
                        default="body")

    parser.add_argument('-s', '--samples', type=int,
                        help='number of files to use for the training. They are chosen \
                        at random.', default=None)

    parser.add_argument('-p', '--parameters', type=str,
                        help="Path to a yaml file containing the training parameters. \
                        Will be copied in the log path", default=None)

    parser.add_argument("-v", "--verify", action=argparse.BooleanOptionalAction,
                        help="Verify the dataset implementation. ")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arg()

    if args.integrate:
        integrate()
        exit()

    elif args.verify:
        verify_ds()
        exit()

    params = parse_param(args.parameters)
    training(params)
