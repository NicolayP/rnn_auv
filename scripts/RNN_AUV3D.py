import torch
torch.autograd.set_detect_anomaly(True)
import pypose as pp
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from utile import read_files, parse_param, save_param, to_euler
from nn_utile import DatasetList3D, AUVRNNDeltaVProxy, AUVRNNDeltaV, AUVStep, AUVTraj, TrajLoss, train

from tqdm import tqdm
import os
import random
random.seed(0)
import warnings

from utile import plot_traj, tdtype, npdtype
import matplotlib.pyplot as plt

import time
from datetime import datetime

import argparse

def get_device(gpu=False, unit=0):
    use_cuda = False
    if gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")
    return torch.device(f"cuda:{unit}" if use_cuda else "cpu")

'''
Runs a given RNN Model with a given input state and action sequence.
'''
def run(model, state, X):
    # TODO: disable all log, just keep the trajectory.
    traj = model(state, X)
    return traj

'''
Load a RNN Model given a checkpoint file.
'''
def load_model(model, ckpt_path):
    ckpt = torch.load(ckpt_path)
    model.load_state_dict(ckpt)
    return model


def create_model(params, device):
    model = AUVTraj(params).to(device)
    return model


def integrate():
    tau = 500
    se3 = True
    device = get_device(gpu=True)
    dir = 'data/csv/sub/'
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    trajs, trajs_plot, Ivels, Bvels, acts, Idvs, Bdvs = [], [], [], [], [], [], []

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

    pred = AUVTraj(se3=True).to(device)
    pred.step = step

    input = torch.concat([trajs[:, :1], Bvels[:, :1]], dim=-1)

    print("input: ", input.shape)
    print("acts:  ", acts.shape)
    pred.step.dv_pred.i = 0
    foo_t, foo_v, foo_dv = pred(input, acts)

    pred.step.dv_pred.i = 0
    s = time.time()
    pred_trajs, pred_vs, pred_dvs = pred(input, acts)
    e = time.time()
    print(f"Prediction time: {e-s}")

    pred_trajs, pred_vs, pred_dvs = pred_trajs.detach().cpu(), pred_vs.detach().cpu(), pred_dvs.detach().cpu()
    trajs, Ivels, Bvels, Idvs, Bdvs = trajs.cpu(), Ivels.cpu(), Bvels.cpu(), Idvs.cpu(), Bdvs.cpu()

    loss_fc = TrajLoss().to(device)

    if not se3:
        pred_trajs_pp = pp.SE3(pred_trajs)
    else:
        pred_trajs_pp = pred_trajs
    trajs_pp = pp.SE3(trajs)
    l = loss_fc(trajs_pp, pred_trajs_pp, Bvels, pred_vs, Bdvs, pred_dvs)

    print(f"Loss: {l} {l.shape}")

    # plotting
    if se3:
        pred_traj_euler = to_euler(pred_trajs[0].data)
    else:
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


def training(params, gpu_number=0):
    nb_files = params["dataset_params"]["samples"]
    data_dir = params["dataset_params"]["dir"]
    stats_file = os.path.join(params["dataset_params"]["dir"], "stats", "stats.yaml")
    stats = parse_param(stats_file)

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
        act_normed=params["dataset_params"]["act_normed"],
        se3=params["model"]["se3"],
        out_normed=params["dataset_params"]["out_normed"],
        stats=stats
    )

    dfs_val = read_files(data_dir, val_files, "val")
    dataset_val = DatasetList3D(
        dfs_val,
        steps=params["dataset_params"]["steps"],
        v_frame=params["dataset_params"]["v_frame"],
        dv_frame=params["dataset_params"]["dv_frame"],
        act_normed=params["dataset_params"]["act_normed"],
        se3=params["model"]["se3"],
        out_normed=params["dataset_params"]["out_normed"],
        stats=stats
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

    device = get_device(True, gpu_number)
    model = AUVTraj(params).to(device)
    if params["dataset_params"]["out_normed"]:
        mean, std = dataset_train.get_stats()
        model.step.set_stats(torch.tensor(mean, dtype=tdtype).to(device),
                            torch.tensor(std, dtype=tdtype).to(device))
    # loss_fc = torch.nn.MSELoss().to(device)
    loss_fc = TrajLoss(params["loss"]["traj"], params["loss"]["vel"], params["loss"]["dv"]).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=params["optim"]["lr"])
    epochs = params["optim"]["epochs"]

    train(ds, model, loss_fc, optim, writer, epochs, device, ckpt_dir)


def verify_ds():
    # Create dataset of 10-15 steps with one or two trajs.
    tau = 10
    v_frame = "body"
    dv_frame = "body"
    data_dir = "data/csv/sub"
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    dfs_verif = read_files(data_dir, files, "verif")
    verif_ds = DatasetList3D(
        dfs_verif,
        steps=tau,
        v_frame=v_frame,
        dv_frame=dv_frame,
        se3=True
    )

    step = AUVStep(v_frame=v_frame, dv_frame=dv_frame)
    pred = AUVTraj()
    pred.step = step

    loss = torch.nn.MSELoss()
    error_traj = torch.zeros(size=(1, tau, 7))
    error_vel = torch.zeros(size=(1, tau, 6))
    error_dv = torch.zeros(size=(1, tau, 6))
    # Create a DV predictor for every entry.

    for data in tqdm(verif_ds):
        x, u, traj, vel, dv = data
        x = torch.tensor(x)[None]
        u = torch.tensor(u)[None]
        traj = torch.tensor(traj)[None]
        vel = torch.tensor(vel)[None]
        dv = torch.tensor(dv)[None]

        dv_pred = AUVRNNDeltaVProxy(dv)
        pred.step.dv_pred = dv_pred
        pred.step.dv_pred.i = 0
        # Integrate the DV predictor
        pred_trajs, pred_vels, pred_dvs = pred(x, u)

        #print("pred_traj", pred_trajs.shape)
        error_traj += loss(pred_trajs, traj)
        error_vel += loss(pred_vels, vel)
        error_dv += loss(pred_dvs, dv)

        pred_trajs = pred_trajs.detach().cpu()
        pred_vels = pred_vels.detach().cpu()
        pred_dvs = pred_dvs.detach().cpu()

        traj = traj.detach().cpu()
        Bvel = vel.detach().cpu()
        Bdvs = dv.detach().cpu()

        pred_traj_euler = to_euler(pred_trajs[0].data)
        traj_euler = to_euler(traj[0])
        # Can investigate every sup prediciton to be sure.
        s_col = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
        plot_traj({"pred": pred_traj_euler, "gt": traj_euler}, s_col, tau, True, title="State")
        v_col = {"u": 0, "v": 1, "w": 2, "p": 3, "q": 4, "r": 5}
        plot_traj({"pred": pred_vels[0], "gt": Bvel[0]}, v_col, tau, True, title="Velocities")
        dv_col = {"Bdu": 0, "Bdv": 1, "Bdw": 2, "Bdp": 3, "Bdq": 4, "Bdr": 5}
        plot_traj({"pred": pred_dvs[0], "gt": Bdvs[0]}, dv_col, tau, True, title="Velocities Deltas")
        plt.show()


    # Compare it to the Expected outcome.

    #not sure about that one
    error_euler = to_euler(error_traj[0])
    s_col = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
    plot_traj({"pred": error_euler}, s_col, tau, True, title="State")
    v_col = {"u": 0, "v": 1, "w": 2, "p": 3, "q": 4, "r": 5}
    plot_traj({"pred": error_vel[0]}, v_col, tau, True, title="Velocities")
    plt.show()


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

    parser.add_argument("-g", "--gpu", type=int,
                        help="select the gpu number, automatically uses the gpu\
                        if available", default=0)

    parser.add_argument("-r", "--run", type=str,
                        help="when enabled, it will run the model specified in the \
                        directory. The directory needs to contain a checkpoint file, \
                        a run file and a parameter file.")

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

    elif args.run:
        path = args.run
        param_file = os.path.join(path, "parameters.yaml")
        param = parse_param(param_file)
        ckpt_path = os.path.join(path, "ckpt.pth")

        device = get_device(True)

        state = torch.zeros(size=(2000, 1, 13)).to(device)
        state[..., 6] = 1.
        seq = torch.zeros(size=(2000, 50, 6)).to(device)
        model = create_model(param, device)
        model = load_model(model, ckpt_path)

        print("Prediction...")
        l = 20
        s = time.time()
        for i in tqdm(range(l)):
            traj = run(model, state, seq)
        e = time.time()
        print(f"Average Prediction Time: {(e-s)/l}")
        print("Output", traj[0].shape)
        exit()

    params = parse_param(args.parameters)
    training(params, args.gpu)
