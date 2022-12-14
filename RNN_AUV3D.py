import torch
torch.autograd.set_detect_anomaly(True)
import pypose as pp
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os
import random
random.seed(0)
import warnings

from scipy.spatial.transform import Rotation as R

npdtype = np.float32
tdtype = torch.float32

from utile import plot_traj
import matplotlib.pyplot as plt

import time


class AUVRNNDeltaVProxy(torch.nn.Module):
    def __init__(self, dv):
        super(AUVRNNDeltaVProxy, self).__init__()
        self._dv = dv
        self.i = 0
    
    def forward(self, x, v, a, h0=None):
        self.i += 1
        return self._dv[:, self.i], None


class AUVRNNDeltaV(torch.nn.Module):
    def __init__(self):
        super(AUVRNNDeltaV, self).__init__()
    
    def forward(self, x, v, a, h0=None):
        pass


class AUVStep(torch.nn.Module):
    def __init__(self, dt=0.1):
        super(AUVStep, self).__init__()
        self.dv_pred = AUVRNNDeltaV()
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
        # Compute the displacement.
        t = v*self.dt
        # Update pose using right \oplus operator as we assume the 
        # velocity to be in body frame.
        x_next = x * pp.se3(t).Exp()
        # Update the velocity vector.
        # Bring the current velocity in the lie algebra
        I_v = x.Adj(v)
        # Add the velocity delta (assumed in world frame)
        I_v_next = I_v + dv
        # Revert the new velocity back into the body frame.
        v_next = x.Inv().Adj(I_v_next)
        return x_next, v_next, dv, h_next


class AUVTraj(torch.nn.Module):
    def __init__(self):
        super(AUVTraj, self).__init__()
        self.step = AUVStep()

    def forward(self, p, v, A):
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
        traj = torch.zeros(size=(k, tau+1, 7)).to(p.device)
        traj[:, 0] = p
        traj_v = torch.zeros(size=(k, tau+1, 6)).to(p.device)
        traj_v[:, 0] = v
        traj_dv = torch.zeros(size=(k, tau+1, 6)).to(p.device)
        
        x = pp.SE3(p).to(p.device)

        for i in range(tau):
            x_next, v_next, dv, h_next = self.step(x, v, A[:, i], h)
            x, v, h = x_next, v_next, h_next
            traj[:, i+1] = x.data
            traj_v[:, i+1] = v
            traj_dv[:, i+1] = dv
        return traj, traj_v, traj_dv


def get_device(gpu=False):
    use_cuda = False
    if gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")
    return torch.device("cuda:0" if use_cuda else "cpu")


def to_euler(traj):
    # assume quaternion representation
    p = traj[:, :3]
    q = traj[:, 3:]
    r = R.from_quat(q)
    e = r.as_euler('xyz')
    return np.concatenate([p, e], axis=-1)


def integrate():
    tau = 300
    device = get_device()
    dir = 'test_data_clean/csv/'
    files = [os.path.join(dir, f) for f in os.listdir(dir)]
    for f in files:
        df = pd.read_csv(f)

        traj = torch.Tensor(df[['x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']].to_numpy())
        traj_plot = torch.Tensor(df[['x', 'y', 'z', 'roll', 'pitch', 'yaw']].to_numpy())
        vels = torch.Tensor(df[['Bu', 'Bv', 'Bw', 'Bp', 'Bq', 'Br']].to_numpy())
        act = torch.Tensor(df[['Fx', 'Fy', 'Fz', 'Tx', 'Ty', 'Tz']].to_numpy())
        Idv = torch.Tensor(df[['Idu', 'Idv', 'Idw', 'Idp', 'Idq', 'Idr']].to_numpy())
        Bdv = torch.Tensor(df[['Bdu', 'Bdv', 'Bdw', 'Bdp', 'Bdq', 'Bdr']].to_numpy())

        dv_pred = AUVRNNDeltaVProxy(Idv[None]).to(device)

        step = AUVStep().to(device)
        step.dv_pred = dv_pred

        pred = AUVTraj().to(device)
        pred.step = step


        s = time.time()
        pred_traj, pred_v, pred_dv = pred(traj[None, 0], vels[None, 0], act[None, :-1])
        e = time.time()
        print(f"Prediction time: {e-s}")

        pred_traj_euler = to_euler(pred_traj.numpy()[0])

        t_dict = {'x': 0, 'y': 1, 'z': 2, 'roll': 3, 'pitch': 4, 'yaw': 5}
        plot_traj({"Traj": traj_plot, "Pred": pred_traj_euler}, t_dict, tau, True)
        v_dict = {'Bu': 0, 'Bv': 1, 'Bw': 2, 'Bp': 3, 'Bq': 4, 'Br': 5}
        plot_traj({"Vel": vels, "Pred_vel": pred_v[0]}, v_dict, tau, True, "Velocity")
        #a_dict = {'Fx': 0, 'Fy': 1, 'Fz': 2, 'Tx': 3, 'Ty': 4, 'Tz': 5}
        #plot_traj({"Act": act}, a_dict, tau, True, "Action")
        dv_dict = {'Idu': 0, 'Idv': 1, 'Idw': 2, 'Idp': 3, 'Idq': 4, 'Idr': 5}
        plot_traj({"Idv": Idv, "Pred_Idv": pred_dv[0]}, dv_dict, tau, True, "Velocity delta")
        
        plt.show()



if __name__ == "__main__":
    integrate()
    