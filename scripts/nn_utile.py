import torch
#torch.autograd.set_detect_anomaly(True)
import pypose as pp
import numpy as np
#from tqdm import tqdm
from utile import tdtype, npdtype, to_euler, gen_imgs_3D
import os

#########################################
#     Network and proxy definitons      #
#########################################
'''
    Proxy for the RNN part of the network. Used to ensure that
    the integration using PyPose is correct.
'''
class AUVRNNDeltaVProxy(torch.nn.Module):
    '''
        Delta veloctiy proxy constructor.

        input:
        ------
            - dv, pytorch tensor. The ground truth velocity delta.
            Shape (k, tau, 6)
    '''
    def __init__(self, dv):
        super(AUVRNNDeltaVProxy, self).__init__()
        self._dv = dv
        self.i = 0

    '''
        Forward function.
        Returns the next ground truth entry. Inputs are only there
        to match the function prototype.

        inputs:
        -------
            - x, that state of the vehicle (not used)
            - v, the velocity of the vehicle (not used)
            - u, the action applied to the vehicle (not used)
            - h0, the vecotr representing the last steps (used for rnn but not here)

        outputs:
        --------
            - dv[:, current-1:current, ...], the current velocity delta.
                shape [k, 1, 6]
            - hNext: set to None.
    '''
    def forward(self, x, v, a, h0=None):
        self.i += 1
        return self._dv[:, self.i-1:self.i], None

'''
    RNN predictor for $\delta v$.

    parameters:
    -----------
        - rnn:
            - rnn_layer: int, The number of rnn layers.
            - rnn_hidden_size: int, Number of hidden units
            - bias: Whether bool, or not to apply bias to the RNN units. (default False)
            - activation: string, The activation function used (tanh or relu)
        - fc:
            - topology: array of ints, each entry indicates the number of hidden units on that
                corresponding layer.
            - bias: bool, Whether or not to apply bias to the FC units. (default False)
            - batch_norm: bool, Whether or not to apply batch normalization. (default False)
            - relu_neg_slope: float, the negative slope of the relu activation.
'''
class AUVRNNDeltaV(torch.nn.Module):
    '''
        RNN network predictinfg the next velocity delta.

        inputs:
        -------
            params: dict, the parameters that define the topology of the network.
            see in class definition.
    '''
    def __init__(self, params=None):
        super(AUVRNNDeltaV, self).__init__()

        self.input_size = 9 + 6 + 6 # rotation matrix + velocities + action. I.E 21
        self.output_size = 6

        # RNN part
        self.rnn_layers = 5
        self.rnn_hidden_size = 1
        rnn_bias = False
        nonlinearity = "tanh"

        # FC part
        topology = [32, 32]
        fc_bias = False
        bn = True
        relu_neg_slope = 0.1

        if params is not None:
            if "rnn" in params:
                if "rnn_layer" in params["rnn"]:
                    self.rnn_layers = params["rnn"]["rnn_layer"]
                if "rnn_hidden_size" in params["rnn"]:
                    self.rnn_hidden_size = params["rnn"]["rnn_hidden_size"]
                if "bias" in params["rnn"]:
                    rnn_bias = params["rnn"]["bias"]
                if "activation" in params["rnn"]:
                    nonlinearity = params["rnn"]["activation"]

            if "fc" in params:
                if "topology" in params["fc"]:
                    topology = params["fc"]["topology"]
                if "bias" in params["fc"]:
                    fc_bias = params["fc"]["bias"]
                if "batch_norm" in params["fc"]:
                    bn = params["fc"]["batch_norm"]
                if "relu_neg_slope" in params["fc"]:
                    relu_neg_slope = params["fc"]["relu_neg_slope"]

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

            fc_layers.append(torch.nn.LeakyReLU(negative_slope=relu_neg_slope))

        layer = torch.nn.Linear(topology[-1], 6, bias=fc_bias)
        fc_layers.append(layer)

        self.fc = torch.nn.Sequential(*fc_layers)
        #self.fc.apply(init_weights)

    '''
        Forward function of the velocity delta predictor.

        inputs:
        -------
            - x, the state of the vehicle. Pypose element. Shape (k, se3_rep)
            - v, the velocity of the vehicle. Pytorch tensor, Shape (k, 6)
            - u, the action applied to the vehicle. Pytorch tensor, Shape (k, 6)
            - h0, the internal state of the rnn unit. Shape (rnn_layers, k, rnn_hiden_size)
                if None, the object will create a new one.

        outputs:
        --------
            - dv, the next velocity delta. Tensor, shape (k, 6, 1)
            - hN, the next rnn internal state. Shape (rnn_layers, k, rnn_hidden_size)
    '''
    def forward(self, x, v, u, h0=None):
        k = x.shape[0]
        r = x.rotation().matrix().flatten(start_dim=-2)
        input_seq = torch.concat([r, v, u], dim=-1)

        if h0 is None:
            h0 = self.init_hidden(k, x.device)

        out, hN = self.rnn(input_seq, h0)
        dv = self.fc(out[:, 0])
        return dv[:, None], hN

    '''
        Helper function to create the rnn internal layer.

        inputs:
        -------
            - k, int, the batch size.
            - device, the device on which to load the tensor.

        outputs:
        --------
            - h0, shape (rnn_layers, k, rnn_hidden_size)
    '''
    def init_hidden(self, k, device):
        return torch.zeros(self.rnn_layers, k, self.rnn_hidden_size, device=device)

'''
    Performs a single integration step using pypose and velocity delta.

    parameters:
    -----------
        - model: dict, entry that contains the NN model definition. See AUVRNNDeltaV.
        - dataset_params:
            - v_frame: string, The frame in which the velocity is expressed, world or body
                default: body
            - dv_frame: string, The frame in which the velocity delta is expressed, world or body
                default: body
'''
class AUVStep(torch.nn.Module):
    '''
        AUVStep Constructor.

        inputs:
        -------
            - params, dict. See object definition above.
            - dt, the integration time.
    '''
    def __init__(self, params=None, dt=0.1):
        super(AUVStep, self).__init__()
        if params is not None:
            self.dv_pred = AUVRNNDeltaV(params["model"])

            if "dataset_params" in params:
                if "v_frame" in params["dataset_params"]:
                    self.v_frame = params["dataset_params"]["v_frame"]
                if "dv_frame" in params["dataset_params"]:
                    self.dv_frame = params["dataset_params"]["dv_frame"]

        else:
            self.dv_pred = AUVRNNDeltaV()
            self.v_frame = "body"
            self.dv_frame = "body"

        self.dt = dt
        self.std = 1.
        self.mean = 0.

    '''
        Predicts next state (x_next, v_next) from (x, v, a, h0)

        inputs:
        -------
            - x, pypose.SE3. The current pose on the SE3 manifold.
                shape [k, 7] (pypose uses quaternion representation)
            - v, torch.Tensor \in \mathbb{R}^{6} \simeq pypose.se3.
                The current velocity. shape [k, 6]
            - u, torch.Tensor the current applied forces.
                shape [k, 6]
            - h0, the hidden state of the RNN network.

        outputs:
        --------
            - x_next, pypose.SE3. The next pose on the SE3 manifold.
                shape [k, 7]
            - v_next, torch.Tensor \in \mathbb{R}^{6} \simeq pypose.se3.
                The next velocity. Shape [k, 6]
            - dv, torch.Tensor The velocity delta. Used for debugging.\
                Warning. dv is unnormed.
            - h_next, torch.Tensor. The next internal representation of
                the RNN.
    '''
    def forward(self, x, v, u, h0=None):
        dv, h_next = self.dv_pred(x, v, u, h0)

        dv_unnormed = dv*self.std + self.mean

        t = pp.se3(self.dt*v).Exp()
        x_next = x * t
        v_next = v + x.Inv().Adj(dv_unnormed)

        return x_next, v_next, dv, h_next                     

    '''
        Set the mean and variance of the input data.
        This will be used to normalize the input data.

        inputs:
        -------
            - mean, tensor, shape (6)
            - std, tensor, shape (6)
    '''
    def set_stats(self, mean, std):
        self.mean = mean
        self.std = std

'''
    Performs full trajectory integration.

    params:
    -------
        - model:
            - se3: bool, whether or not to use pypose.
            - for other entries look at AUVStep and AUVRNNDeltaV.
'''
class AUVTraj(torch.nn.Module):
    '''
        Trajectory generator objects.

        inputs:
        -------
            - params: see definnition above.
    '''
    def __init__(self, params=None):
        super(AUVTraj, self).__init__()
        self.step = AUVStep(params)
        if params is not None:
            self.se3 = params["model"]["se3"]
        else:
            self.se3 = True

    '''
        Generates a trajectory using a Inital State (pose + velocity) combined
        with an action sequence.

        inputs:
        -------
            - x, torch.Tensor. The pose of the system with quaternion representation and the velocity.
                shape [k, 7+6]
            - U, torch.Tensor. The action sequence appliedto the system.
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
    def forward(self, x, U):
        k = U.shape[0]
        tau = U.shape[1]
        h = None
        p = x[..., :7]
        v = x[..., 7:]
        traj = torch.zeros(size=(k, tau, 7)).to(p.device)
        traj = pp.SE3(traj)
        traj_v = torch.zeros(size=(k, tau, 6)).to(p.device)
        traj_dv = torch.zeros(size=(k, tau, 6)).to(p.device)
        
        x = pp.SE3(p).to(p.device)
        for i in range(tau):
            x_next, v_next, dv, h_next = self.step(x, v, U[:, i:i+1], h)
            x, v, h = x_next, v_next, h_next
            traj[:, i:i+1] = x
            traj_v[:, i:i+1] = v
            traj_dv[:, i:i+1] = dv
        return traj, traj_v, traj_dv

'''
    Compute the Left-Geodesic loss between two SE(3) poses.
'''
class GeodesicLoss(torch.nn.Module):
    '''
        GeodesicLoss constructor
    '''
    def __init__(self):
        super(GeodesicLoss, self).__init__()

    '''
        inputs:
        -------
            - X1 pypose.SE3. The first pose.
            - X2 pypose.SE3. The second pose.

        outputs:
        --------
            - Log(X1 + X2^{-1})^{2}
    '''
    def forward(self, X1, X2):
        d = (X1 * X2.Inv()).Log()
        square = torch.pow(d, 2)
        return square

'''

'''
class TrajLoss(torch.nn.Module):
    '''
        Trajectory loss consstructor.

        inputs:
        -------
            - alpha: float, weight for trajectory loss.
            - beta: float, weight for velocity loss.
            - gamma: float, weight for $\delta V$ loss.
    '''
    def __init__(self, alpha=1., beta=0., gamma=0.):
        super(TrajLoss, self).__init__()
        self.l2 = torch.nn.MSELoss()
        self.geodesic = GeodesicLoss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        pass

    '''
        Returns true if beta > 0.
    '''
    def has_v(self):
        return self.beta > 0.

    '''
        Returns true if gamma > 0.
    '''
    def has_dv(self):
        return self.gamma > 0.

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
    def forward(self, traj1, traj2, v1=None, v2=None, dv1=None, dv2=None, split=False):
        if split:
            return self.split_loss(traj1, traj2, v1, v2, dv1, dv2)
        return self.loss(traj1, traj2, v1, v2, dv1, dv2)

    '''
        Computes trajectory, velocity and $\Delta V$ loss split accross each dimesnions.

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

        outputs:
        --------
            t_l: torch.tensor, trajectory loss
                shape [6]
            v_l: torch.tensor, velocity loss
                shape [6]
            dv_l: torch.tensor, delta velocity loss
                shape [6]
    '''
    def split_loss(self, t1, t2, v1, v2, dv1, dv2):
        # only used for logging and evaluating the performances.
        t_l = self.geodesic(t1, t2).mean((0, 1))
        v_l = torch.pow(v1 - v2, 2).mean((0, 1))
        dv_l = torch.pow(dv1 - dv2, 2).mean((0, 1))
        return t_l, v_l, dv_l

    '''
        Computes trajectory, velocity and $\Delta V$ loss.

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

        outputs:
        --------
            loss: the full trajectory loss.
    '''
    def loss(self, t1, t2, v1, v2, dv1, dv2):
        t_l = self.geodesic(t1, t2).mean()
        v_l = self.l2(v1, v2).mean()
        dv_l = self.l2(dv1, dv2).mean()
        return self.alpha*t_l + self.beta*v_l + self.gamma*dv_l


# DATASET FOR 3D DATA
class DatasetList3D(torch.utils.data.Dataset):
    '''
        Dataset Constructor.

        inputs:
        -------
            - data_list: List, A list of pandas dataframe representing trajectories.
            - steps: Int, The number of steps to use for prediction.
            - v_frame: String, The frame in which the velocity is represented (world or body)
            - dv_frame: String, The frame in whicch the velocity delta is represented (world or body)
            - rot: String, the representation used for rotations. (only quat supported at the moment.)
            - act_normed: Bool, whether or not to normalize the action before feeing them to the network.
            - se3: Bool, whether or not to use pypose as underlying library for the pose representation.
            - out_normed: Bool, whether or not to normalize the targets.
            - stats: dict with entries:
                - std:
                    - world_norm: list of floats. Shape (6)
                    - body_norm: list of floats. Shape (6)
                - mean:
                    - world_norm: list of floats. Shape (6)
                    - body_norm: list of floats. Shape (6)
    '''
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

    '''
        returns the number of samples in the dataset.
    '''
    def __len__(self):
        return self.len

    '''
        get a sample at a specific index.

        inputs:
        -------
            - idx, int < self.__len__().

        outputs:
        --------
            - x, the state of the vehicle (pose and velocity)
                shape (1, 7+6)
            - u, The actions applied to the vehicle. Shape (steps, 6)
            - traj, The resulting trajectory. Shape (steps, 7)
            - vel, The resulting velocity profiles, shape (steps, 6)
            - dv, The normalized velocity delta prrofiles, shape (steps, 6)
    '''
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

        traj = pp.SE3(traj)

        return x, u, traj, vel, dv

    '''
        Returns the number of trajectories in the dataset.
    '''
    @property
    def nb_trajs(self):
        return len(self.data_list)
    
    '''
        Get the traj at a specific index ind the dataset.

        inputs:
        -------
            - idx, int, the trajectory index.

        outputs:
        --------
            - trajectory, shape (tau, 7+6)
    '''
    def get_traj(self, idx):
        if idx >= self.nb_trajs:
            raise IndexError
        return self.data_list[idx][self.x_labels].to_numpy()
    
    '''
        internal function that creats bins to compute the number
        of samples in the dataset.
    '''
    def create_bins(self):
        bins = [0]
        cummul = 0
        for s in self.samples:
            cummul += s
            bins.append(cummul)
        return bins

    '''
        get all the trajectories from the dataset. Only works if all
        the trajs in the dataset have the same length.

        inputs:
        -------
            - None

        outputs:
        --------
            - trajs, shape (nb_traj, tau, se3_rep)
            - vels, shape (nb_traj, tau, 6)
            - dvs, shape (nb_traj, tau, 6)
            - actions, shape (nb_traj, tau, 6)
    '''
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

    '''
        Get the mean and std of the velocity delta.

        outputs:
        --------
            - mean, torch.tensor, shape [6]
            - std, torch.tensor, shape [6]
    '''
    def get_stats(self):
        return self.mean, self.std

'''
    Inits the weights of the neural network.

    inputs:
    -------
        - m the neural network layer.
'''
def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)


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
def traj_loss(dataset, model, loss, tau, writer, step, device, mode="train", plot=False):
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

# TRAINING AND VALIDATION
'''
'''
def val_step(dataloader, model, loss, writer, epoch, device):
    torch.autograd.set_detect_anomaly(True)
    size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Val: {epoch}", ncols=200, colour="red", leave=False)
    model.eval()
    for batch, data in t:
        X, U, traj, vel, dv = data
        X, U = X.to(device), U.to(device)
        traj, vel, dv = traj.to(device), vel.to(device), dv.to(device)

        pred, pred_vel, pred_dv = model(X, U)
        l = loss(traj, pred, vel, pred_vel, dv, pred_dv)

        if writer is not None:
            writer.add_scalar("val/loss", l, epoch*size+batch*len(X))

    # Trajectories generation for validation
    tau = [50]
    traj_loss(dataloader.dataset, model, loss, tau, writer, epoch, device, "val", True)

'''
'''
def train_step(dataloader, model, loss, optim, writer, epoch, device):
    #print("\n", "="*5, "Training", "="*5)
    torch.autograd.set_detect_anomaly(True)
    size = len(dataloader.dataset)
    t = tqdm(enumerate(dataloader), desc=f"Epoch: {epoch}", ncols=200, colour="red", leave=False)
    model.train()
    for batch, data in t:
        X, U, traj, vel, dv = data
        X, U, traj, vel, dv = X.to(device), U.to(device), traj.to(device), vel.to(device), dv.to(device)

        pred, pred_v, pred_dv = model(X, U)

        optim.zero_grad()
        l = loss(traj, pred, vel, pred_v, dv, pred_dv)
        l.backward()
        optim.step()

        if writer is not None:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    writer.add_histogram("train/" + name, param, epoch*size+batch*len(X))
            l_split = loss(traj, pred, vel, pred_v, dv, pred_dv, split=True)
            name = [["x", "y", "z", "vec_x", "vec_y", "vec_z"],
                ["u", "v", "w", "p", "q", "r"],
                ["du", "dv", "dw", "dp", "dq", "dr"]]
            for d in range(6):
                for i in range(3): 
                    writer.add_scalar("train/split-loss-" + name[i][d], l_split[i][d], epoch*size+batch*len(X))
            writer.add_scalar("train/loss", l, epoch*size+batch*len(X))

    return l.item(), batch*len(X)

'''
'''
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
        if (e % ckpt_steps == 0) and ckpt_dir is not None:
            tau=[50]
            traj_loss(ds[0].dataset, model, loss_fc, tau, writer, e, device, "train", True)
            val_step(ds[1], model, loss_fc, writer, e, device)

            if ckpt_steps > 0:
                tmp_path = os.path.join(ckpt_dir, f"step_{e}.pth")
                torch.save(model.state_dict(), tmp_path)

        l, cur = train_step(ds[0], model, loss_fc, optim, writer, e, device)
        t.set_postfix({"loss": f"Loss: {l:>7f} [{cur:>5d}/{size:>5d}]"})

        if writer is not None:
            writer.flush()
