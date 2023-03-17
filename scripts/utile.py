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
'''
    Reads csv files from a directory.
    
    input:
    ------
        - data_dir, string. relative or absoulte path to the directory
        containing the csv files.
        - files, list of string. List containing the names of all the files
        the need to be loaded.
        - type, string. Decorator for tqdm.

    output:
    -------
        - dfs, list of dataframe containing the loaded csv files.
'''
def read_files(data_dir, files, type="train"):
    dfs = []
    for f in tqdm(files, desc=f"Dir {type}", ncols=150, colour="blue"):
        csv_file = os.path.join(data_dir, f)
        df = pd.read_csv(csv_file)
        df = df.astype(npdtype)
        dfs.append(df)
    return dfs

'''
    Plots trajectories with euler representation and velocity profiles from 
    the trajectory and velocity dictionaries respectively.
    The trajectories are plotted with length tau.

    input:
    ------
        - t_dict, dictionnary. Entries are "plotting-label": trajectory. The key will be used as
        label for the plots. The trajectory need to have the following entries [x, y, z, roll, pitch, yaw].
        - v_dict, dictionnary. Entries are "plotting-label": velocities. The key will be used as
        label for the plots. The veloties need to have the following entries [u, v, w, p, q, r].
        - dv_dict, dictionnary (default, None). Entries are "plotting-label": \detla V. The key will be used as
        label for the plots. The \delta V need to have the following entries [\delta u, \delta v, \delta w, \delta p, \delta q, \delta r].
        - tau, int. The number of points to plot

    output:
    -------
        - image that can be plotted or send to tensorboard. Returns tuple (trajectory_img, velocity_img).
        if dv_dict is not None, returns (trajectory_img, velocity_img, delta_v_img)
'''
def gen_imgs_3D(t_dict, v_dict, dv_dict=None, tau=100):
    plotState={"x(m)":0, "y(m)": 1, "z(m)": 2, "roll(rad)": 3, "pitch(rad)":4, "yaw(rad)": 5}
    plotVels={"u(m/s)":0, "v(m/s)": 1, "w(m/s)": 2, "p(rad/s)": 3, "q(rad/s)": 4, "r(rad/s)": 5}
    plotDVels={"du(m/s)":0, "dv(m/s)": 1, "dw(m/s)": 2, "dp(rad/s)": 3, "dq(rad/s)": 4, "dr(rad/s)": 5}
    t_imgs = []
    v_imgs = []
    if dv_dict is not None:
        dv_imgs = []
    for t in tau:
        t_imgs.append(plot_traj(t_dict, plotState, t, title="State evolution"))
        v_imgs.append(plot_traj(v_dict, plotVels, t, title="Velcoity Profiles"))
        if dv_dict is not None:
            dv_imgs.append(plot_traj(dv_dict, plotDVels, t, title="Delta V"))

    if dv_dict is not None:
        return t_imgs, v_imgs, dv_imgs

    return t_imgs, v_imgs

'''
    Plots trajectories from a dictionnary.

    input:
    ------
        - traj_dict, dict. Entries are "plotting-label": trajectory. The key will be used as
        label for the plots.
        - plot_cols, dict. Entires are "axis-name": index-in-trajectory. This matches the trajectory
        from traj_dict.
        - tau, int. The number of steps to plot.
        - fig, bool (default false). If true, returns the matplotlib figure that can be shown with plt.show().
        - title, string. The title of the graph.
        - save, bool. (default false) If true, save the image in a dir called Img.

    output:
    -------
        - if fig == True, returns the matplotlib figure.
        - if fig == False, returns a np.array containing the RBG values of the image.
'''
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

'''
    Converts a trajectory using quaternion representation to euler 'xyz' angle representation.
    
    input:
    ------
        - traj, numpy array. The trajectory with quaternion representation. It assumes that the quaternion
        is represented with entry index 3-7.

    output:
    -------
        - traj_euler, numpy array. The same trajectory with euler representation.
'''
def to_euler(traj):
    # assume quaternion representation
    p = traj[..., :3]
    q = traj[..., 3:]
    r = R.from_quat(q)
    e = r.as_euler('xyz')
    return np.concatenate([p, e], axis=-1)


def traj_loss(dataset, model, loss, tau, writer, step, device, mode="train", plot=False):
    '''
        Computes the loss on an entire trajectory. If plot is true, it also plots the
        predicted trajecotry for different horizons.

        input:
        ------
            - dataset: torch.utils.data.Dataset with a methods called get_trajs() that
            returns full trajectory contained in the dataset.
            - model: the dynamical model used for predicitons.
            - loss: torch.function, the loss function used to measure the performance of the model.
            - tau: list of ints, the horizons we want to measure the performance on in increasing order.
            - writer: torch.summarywriter. Writer used to log the data
            - step: the current step in the training process used for logging.
            - device: string, the device to run the model on.
            - mode: string (default: "train") or val. Defines the mode in which this funciton is called.
            - plot: bool (default: False) if true, plots the first trajectory of the dataset as well as
                the on predicted by the model.
    '''
    gt_trajs, gt_vels, gt_dv, aciton_seqs = dataset.get_trajs()
    x_init = gt_trajs[:, 0:1].to(device)
    v_init = gt_vels[:, 0:1].to(device)
    A = aciton_seqs[:, :tau[-1]].to(device)
    init = torch.concat([x_init.data, v_init], dim=-1)

    pred_trajs, pred_vels, pred_dvs = model(init, aciton_seqs.to(device))


    losses = [loss(
            pred_trajs[:, :h], gt_trajs[:, :h].to(device),
            pred_vels[:, :h], gt_vels[:, :h].to(device),
            pred_dvs[:, :h], gt_dv[:, :h].to(device)
        ) for h in tau]
    losses_split = [[loss(
            pred_trajs[:, :h], gt_trajs[:, :h].to(device),
            pred_vels[:, :h], gt_vels[:, :h].to(device),
            pred_dvs[:, :h], gt_dv[:, :h].to(device), split=True
        )] for h in tau]

    name = [["x", "y", "z", "vec_x", "vec_y", "vec_z"],
            ["u", "v", "w", "p", "q", "r"],
            ["du", "dv", "dw", "dp", "dq", "dr"]]
    if writer is not None:
        for i, (l, l_split, t) in enumerate(zip(losses, losses_split, tau)):
            writer.add_scalar(f"{mode}/{t}-multi-step-loss-all", l, step)
            for d in range(6):
                for j in range(3):
                    writer.add_scalar(f"{mode}/{t}-multi-step-loss-{name[j][d]}", l_split[i][j][d], step)

    if not plot:
        return

    t_dict = {
        "model": to_euler(pred_trajs[0].detach().cpu().data),
        "gt": to_euler(gt_trajs[0].data)
    }

    v_dict = {
        "model": pred_vels[0].detach().cpu(),
        "gt": gt_vels[0]
    }

    dv_dict = {
        "model": pred_dvs[0].detach().cpu(),
        "gt": gt_dv[0]
    }

    t_imgs, v_imgs, dv_imgs = gen_imgs_3D(t_dict, v_dict, dv_dict, tau=tau)

    for t_img, v_img, dv_img, t in zip(t_imgs, v_imgs, dv_imgs, tau):
        writer.add_image(f"{mode}/traj-{t}", t_img, step, dataformats="HWC")
        writer.add_image(f"{mode}/vel-{t}", v_img, step, dataformats="HWC")
        writer.add_image(f"{mode}/dv-{t}", dv_img, step, dataformats="HWC")

'''
    Reads a yaml file and returns the matching dictionnary.
    
    input:
    ------
        - file, string. String to the yaml file.

    output:
    -------
        - dict, the associated dictionnary.
'''
def parse_param(file):
    with open(file) as file:
        conf = yaml.load(file, Loader=yaml.FullLoader)
    return conf

'''
    Saves a dictionnary to a yaml file.

    input:
    ------
        - path, string. Filename.
        - params, dict. The dictionnary to be saved.
'''
def save_param(path, params):
    with open(path, "w") as stream:
        yaml.dump(params, stream)