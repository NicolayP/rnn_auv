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
    model.eval()
    for batch, data in t:
        X, U, Y = data
        X, U, Y = X.to(device), U.to(device), Y.to(device)
        pred, pred_dv = model(X, U)
        l = loss(pred_dv, Y)

        if writer is not None:
            writer.add_scalar("val-loss/", l, epoch*size+batch*len(X))
    
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
        X, U, dv, traj = data
        X, U, dv, traj = X.to(device), U.to(device), dv.to(device), traj.to(device)

        pred, pred_v, pred_dv = model(X, U)
        optim.zero_grad()
        l = loss(traj, pred)
        l.backward()
        optim.step()

        if writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram("train/" + name, param, epoch*size+batch*len(X))
                for dim in range(6):
                    loss_dim = loss(pred_dv[..., dim], Y[..., dim])
                    writer.add_scalar("dv-split-loss/" + str(dim), loss_dim, epoch*size+batch*len(X))
                writer.add_scalar("train-loss/", l, epoch*size+batch*len(X))

    return l.item(), batch*len(X)


def parse_param(file):
    with open(file) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf


def save_param(path, params):
    with open(path, "w") as stream:
        yaml.dump(params, stream)