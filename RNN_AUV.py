import torch
torch.autograd.set_detect_anomaly(True)
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from utile import ToSE2Mat, SE2Int, SE2Adj, SE2Inv, FlattenSE2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from tqdm import tqdm
import os
import random
import warnings
random.seed(0)

npdtype = np.float32


class AUVRNN2DDeltaV(torch.nn.Module):
    def __init__(self, bias=False, rnn_hidden_size=5, rnn_layers=1):
        '''
            hist, the number of past states to feed in the network.
            sDim = (x, y, theta, u, v, r)
            u, v, r in body farme and normaized between [-1, 1].
            aDim = (Fx, Fy, Tz) forces in body frame and normalized between [-1, 1].
        '''
        super(AUVRNN2DDeltaV, self).__init__()
        self.hist, self.sDim, self.aDim = 1, 6, 3
        self.input_size = self.sDim - 3 + self.aDim #remove x, y, theta
        self.output_size = self.sDim - 3
        self.rnn_layer = rnn_layers
        self.rnn_hidden_size = rnn_hidden_size

        self.dt = 0.1
        self.rnn = torch.nn.RNN(
            input_size=self.input_size*self.hist,
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

    def forward(self, s, a, h0=None):
        k = s.shape[0]
        input = torch.concat([s[:, :, 3:], a], dim=-1)

        in_shape = input.shape
        if in_shape[1] != self.hist:
            raise ValueError(f"the number of previous step (shape[1]) {in_shape[1]} isn't matching \
                the expected one {self.hist}")

        input = input.view(in_shape[0], 1, -1)

        if h0 is None:
            h0 = self.init_hidden(k, input.device)
        out, hN = self.rnn(input, h0)

        dv = self.fc(out)[:, 0]
        return dv, hN

    def init_hidden(self, k, device):
        return torch.Tensor(np.zeros(shape=(self.rnn_layer, k, self.rnn_hidden_size))).to(device)


class AUVRNN2Dstep(torch.nn.Module):
    def __init__(self, dt=0.1):
        super(AUVRNN2Dstep, self).__init__()
        self.dv_pred = AUVRNN2DDeltaV()
        self.to_mat = ToSE2Mat()
        self.int = SE2Int()
        self.adj = SE2Adj()
        self.inv = SE2Inv()
        self.flat = FlattenSE2()
        self.dt = dt

    def forward(self, s, a, h0=None):
        dv, h = self.dv_pred(s, a, h0)
        #print("\n", "="*5, "Step", "="*5)
        #print("s:      ", s.shape)
        #print("a:      ", a.shape)
        #print("dv:     ", dv.shape)
        v = s[:, -1, -3:] # In body frame
        t = v*self.dt
        #print("v:      ", v)
        #print("t:      ", t)

        M_cur = self.to_mat(s[:, -1, :3])
        #print("M_cur:  \n", M_cur)
        M_next = self.int(M_cur, t)
        #print("M_next: \n", M_next)
        M_next_inv = self.inv(M_next)
        #print("M_inv:  \n", M_next_inv)
        adj = self.adj(M_cur)
        #print("adj:    \n", adj)
        #print("v:      ", v)
        v_i = (adj @ v[..., None])[..., 0] # In sigma algebra
        #print("v_i:    ", v_i)
        v_i_next = v_i + dv
        #print("v_i_next: ", v_i_next)
        v_next = (self.adj(M_next_inv) @ v_i_next[..., None])[..., 0]
        #print("v_next:  ", v_next)

        s_next = self.flat(M_next, v_next)
        #print("s_next:  ", s_next)
        return s_next, dv, h


class AUVRNN2D(torch.nn.Module):
    def __init__(self, bias=False):
        super(AUVRNN2D, self).__init__()
        self.step = AUVRNN2Dstep()

    def forward(self, s, A):
        '''
            Generates a trajectory using a inital state
            and an action sequence.

            inputs:
            -------
                - s, torch.Tensor: the state of the system.
                    shape [k, h, 6]
                - A, torch.Tensor: the action sequence applied
                    on the syste. Shape [k, Tau, 3]

            outputs:
            --------
                - traj, torch.Tensor: the generated trajectory.
                    shape [k, tau, 6]
                - traj_dv, torch.Tensor: the generated delta velocities.
                    shape [k, tau, 3]

        '''
        print("\n", "="*7, "Traj Pred", "="*7)
        print("s: ", s.shape)
        print("A: ", A.shape)
        k = A.shape[0]
        tau = A.shape[1]
        h = None
        traj = torch.zeros(size=(k, tau, 6)).to(s.device)
        traj_dv = torch.zeros(size=(k, tau, 3)).to(s.device)
        for i in range(tau):
            s_next, dv, h_next = self.step(s, A[:, i:i+1], h)
            s, h = s_next, h_next
            traj[:, i:i+1] = s
            traj_dv[:, i] = dv
        return traj, traj_dv


class DatasetList2D(torch.utils.data.Dataset):
    def __init__(self, data_list, steps=1, history=1):
        super(DatasetList2D, self).__init__()
        self.data_list = data_list
        self.s = steps
        self.h = history
        self.pos = ['x', 'y']
        self.rot = ['yaw']
        self.lin_vel = ['u', 'v']
        self.ang_vel = ['r']
        self.x_labels = self.pos + self.rot + self.lin_vel + self.ang_vel
        self.y_labels = ['du', 'dv', 'dr']
        self.u_labels = ['Fx', 'Fy', 'Tz']

        self.samples = [traj.shape[0] - self.h - self.s + 1 for traj in data_list]
        self.len = sum(self.samples)
        self.bins = self.create_bins()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        i = (np.digitize([idx], self.bins)-1)[0]
        traj = self.data_list[i]
        j = idx - self.bins[i]
        sub_frame = traj.iloc[j:j+self.s+self.h]
        x = sub_frame[self.x_labels].to_numpy()
        x = x[:self.h]

        u = sub_frame[self.u_labels].to_numpy()
        u = u[:self.h+self.s-1]

        y = sub_frame[self.y_labels].to_numpy()
        y = y[self.h:self.h+self.s]
        return x, u, y

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


def read_files(data_dir, files, type="train"):
    dfs = []
    for f in tqdm(files, desc=f"Dir {type}", ncols=150, colour="blue"):
        csv_file = os.path.join(data_dir, f)
        df = pd.read_csv(csv_file)
        df = df.astype(npdtype)
        dfs.append(df)
    return dfs


def get_device(gpu=False):
    use_cuda = False
    if gpu:
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")
    return torch.device("cuda:0" if use_cuda else "cpu")


def train(ds, model, loss_fc, optim, writer, epochs, device, ckpt_dir=None, ckpt_steps=2):
    if writer is not None:
        s = torch.Tensor(np.zeros(shape=(1, 1, 6))).to(device)
        A = torch.Tensor(np.zeros(shape=(1, 10, 3))).to(device)
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


def gen_imgs(t_dict, tau):
    plotState={"x(m)":0, "y(m)": 1, "yaw(rad)": 2, "u (m/s)": 3, "v (m/s)":4, "r (rad/s)": 5}
    imgs = []
    for t in tau:
        imgs.append(plot_traj(t_dict, plotState, t))
    return imgs


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
        fig_state.savefig(title + ".png")

    if fig:
        return fig_state

    canvas = FigureCanvas(fig_state)
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig_state.canvas.get_width_height()[::-1] + (3,))
    plt.close('all')
    return img


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
    x_init, A = torch.Tensor(gt_trajs[0][None, 0:1, ...]).to(device), torch.Tensor(action_seqs[0][None, ...]).to(device)
    pred_trajs, pred_trajs_dv = model(x_init, A)
    tau = [10, 20, 30, 40, 50]
    t_dict = {"model": pred_trajs[0].detach().cpu(), "gt": gt_trajs[0]}
    t10_i, t20_i, t30_i, t40_i, t50_i = gen_imgs(t_dict, tau)
    t10_l, t20_l, t30_l, t40_l, t50_l = [loss(pred_trajs_dv[0, :h], torch.Tensor(gt_trajs_dv[0][:h]).to(device)) for h in tau]
    t10_l_u, t10_l_v, t10_l_r = [loss(pred_trajs_dv[0, :10, dim], torch.Tensor(gt_trajs_dv[0][:10, dim]).to(device)) for dim in range(3)]
    t20_l_u, t20_l_v, t20_l_r = [loss(pred_trajs_dv[0, :20, dim], torch.Tensor(gt_trajs_dv[0][:20, dim]).to(device)) for dim in range(3)]
    t30_l_u, t30_l_v, t30_l_r = [loss(pred_trajs_dv[0, :30, dim], torch.Tensor(gt_trajs_dv[0][:30, dim]).to(device)) for dim in range(3)]
    t40_l_u, t40_l_v, t40_l_r = [loss(pred_trajs_dv[0, :40, dim], torch.Tensor(gt_trajs_dv[0][:40, dim]).to(device)) for dim in range(3)]
    t50_l_u, t50_l_v, t50_l_r = [loss(pred_trajs_dv[0, :50, dim], torch.Tensor(gt_trajs_dv[0][:50, dim]).to(device)) for dim in range(3)]


    writer.add_image("traj-10", t10_i, epoch, dataformats="HWC")
    writer.add_image("traj-20", t20_i, epoch, dataformats="HWC")
    writer.add_image("traj-30", t30_i, epoch, dataformats="HWC")
    writer.add_image("traj-40", t40_i, epoch, dataformats="HWC")
    writer.add_image("traj-50", t50_i, epoch, dataformats="HWC")

    writer.add_scalar("Multi-step-loss-t10/all", t10_l, epoch)
    writer.add_scalar("Multi-step-loss-t10/u", t10_l_u, epoch)
    writer.add_scalar("Multi-step-loss-t10/v", t10_l_v, epoch)
    writer.add_scalar("Multi-step-loss-t10/r", t10_l_r, epoch)

    writer.add_scalar("Multi-step-loss-t20/all", t20_l, epoch)
    writer.add_scalar("Multi-step-loss-t10/u", t20_l_u, epoch)
    writer.add_scalar("Multi-step-loss-t10/v", t20_l_v, epoch)
    writer.add_scalar("Multi-step-loss-t10/r", t20_l_r, epoch)

    writer.add_scalar("Multi-step-loss-t30/all", t30_l, epoch)
    writer.add_scalar("Multi-step-loss-t10/u", t30_l_u, epoch)
    writer.add_scalar("Multi-step-loss-t10/v", t30_l_v, epoch)
    writer.add_scalar("Multi-step-loss-t10/r", t30_l_r, epoch)

    writer.add_scalar("Multi-step-loss-t40/all", t40_l, epoch)
    writer.add_scalar("Multi-step-loss-t10/u", t40_l_u, epoch)
    writer.add_scalar("Multi-step-loss-t10/v", t40_l_v, epoch)
    writer.add_scalar("Multi-step-loss-t10/r", t40_l_r, epoch)

    writer.add_scalar("Multi-step-loss-t50/all", t50_l, epoch)
    writer.add_scalar("Multi-step-loss-t10/u", t50_l_u, epoch)
    writer.add_scalar("Multi-step-loss-t10/v", t50_l_v, epoch)
    writer.add_scalar("Multi-step-loss-t10/r", t50_l_r, epoch)


def learning():
    data_dir = "data/clean_csv/"
    #data_dir = "test_data_clean/csv/"
    dir_name = os.path.basename(data_dir)
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    random.shuffle(files)

    # split train and val in 70-30 ratio
    train_size = int(0.7*len(files))

    train_files = files[:train_size]
    val_files = files[train_size:]

    print("Data size:  ", len(files))
    print("Train size: ", len(train_files))
    print("Val size:   ", len(val_files))

    dfs_train = read_files(data_dir, train_files, "train")
    dataset_train = DatasetList2D(dfs_train, steps=10)

    dfs_val = read_files(data_dir, val_files, "val")
    dataset_val = DatasetList2D(dfs_val, steps=10)


    train_params = {
        'batch_size': 2048,
        'shuffle': True,
        'num_workers': 8
    }

    ds = (
        torch.utils.data.DataLoader(
            dataset_train,
            **train_params
        ),
        torch.utils.data.DataLoader(
            dataset_val,
            **train_params
        )
    )

    log_path = "train_log/"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)

    ckpt_dir = "train_ckpt/"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    device = get_device(True)
    model=AUVRNN2D().to(device)
    loss_fc = torch.nn.MSELoss().to(device)
    optim = torch.optim.Adam(model.parameters())
    epochs = 20

    train(ds, model, loss_fc, optim, writer, epochs, device, ckpt_dir)


def investigate():
    device=get_device(False)
    model=AUVRNN2D().to(device)
    # x=0.1m, y=-0.2m, yaw=45 deg, u=0.1m/s, v=0.01m/s, r=-18deg/s
    s = torch.tensor(np.array([[[0.1, -0.2, np.pi/4, 0.1, 0.01, -np.pi/10]]], dtype=npdtype)).to(device)
    s = torch.tensor(np.array([[[1.7771606e-01, 6.2648498e-02, 2.8949225e+00, 6.7778997e-04, 2.6123118e-05, -1.2107605e-04]]], dtype=npdtype)).to(device)
    # Fx=0,01N, Fy=0.0N, Tz=-1Nm
    a = torch.tensor(np.array([[[0.01, 0.0, -1.]]], dtype=npdtype)).to(device)
    a = torch.tensor(np.array([[[0.01, 0.0, -1.]]], dtype=npdtype)).to(device)
    print("S:      ", s.shape)
    print("A:      ", a.shape)
    s = model(s, a)
    print("S next: ", s.shape)


class FakeDv(torch.nn.Module):
    def __init__(self, dv, dt=0.1):
        super(FakeDv, self).__init__()
        self.gt_dv = dv
        self.i = 0
    
    def forward(self, s, a, h0):
        self.i += 1
        return self.gt_dv[:, self.i], None


def integrate():
    tau = 100
    device = get_device()
    traj_file = "test_bag_dv/new/run0.csv"
    df = pd.read_csv(traj_file)

    traj = torch.Tensor(df[['x', 'y', 'yaw', 'u', 'v', 'r']].to_numpy()).to(device)
    act = torch.Tensor(df[['Fx', 'Fy', 'Tz']].to_numpy()).to(device)
    traj_dv = torch.Tensor(df[['du', 'dv','dr']].to_numpy()).to(device)

    dv_pred = FakeDv(traj_dv[None]).to(device)

    step = AUVRNN2Dstep().to(device)
    step.dv_pred = dv_pred

    pred = AUVRNN2D().to(device)
    pred.step = step

    t, t_dv = pred(traj[None, 0:1], act[None, 0:tau])

    t = t[0]
    gt_t = traj[0:tau]

    t_dv = t_dv[0]    
    gt_t_dv = traj_dv[1:tau+1]

    print("t:        ", t.shape)
    print("gt_t:     ", gt_t.shape)

    print("t_dv:     ", t_dv.shape)
    print("gt_t_dv:  ", gt_t_dv.shape)

    print("dv diff:  ", (t_dv - gt_t_dv).sum())

    print("x diff:   ", (torch.abs(t[:, 0] - gt_t[:, 0])).sum())
    print("u diff:   ", (torch.abs(t[:, 3] - gt_t[:, 3])).sum())

    print("y diff:   ", (torch.abs(t[:, 1] - gt_t[:, 1])).sum())
    print("v diff:   ", (torch.abs(t[:, 4] - gt_t[:, 4])).sum())

    print("yaw diff: ", (torch.abs(t[:, 2] - gt_t[:, 2])).sum())
    print("r diff:   ", (torch.abs(t[:, 5] - gt_t[:, 5])).sum())

    t_dict = {"gt": gt_t, "t": t}
    img = gen_imgs(t_dict, [tau])[0]

    plt.imshow(img)
    plt.show()


def inspect_bag():
    traj_files = ["test_bag_dv/old/run0.csv", "test_bag_dv/new/run0.csv"]


    tau = 100
    device = get_device()
    dfs = [pd.read_csv(f) for f in traj_files]
    trajs_np = np.array([df[['x', 'y', 'yaw', 'u', 'v', 'r']].to_numpy() for df in dfs])
    trajs = [torch.Tensor(df[['x', 'y', 'yaw', 'u', 'v', 'r']].to_numpy()).to(device) for df in dfs]
    acts_np = np.array([df[['Fx', 'Fy', 'Tz']].to_numpy() for df in dfs])
    acts = [torch.Tensor(df[['Fx', 'Fy', 'Tz']].to_numpy()).to(device) for df in dfs]
    trajs_dv_np = np.array([df[['du', 'dv','dr']].to_numpy() for df in dfs])
    trajs_dv = [torch.Tensor(df[['du', 'dv','dr']].to_numpy()).to(device) for df in dfs]

    print("trajs_dv: ", trajs_dv_np.shape)
    print("act:      ", acts_np.shape)
    #concat dv and act:
    dv_act = np.concatenate([trajs_dv_np, acts_np], axis=-1)

    print("dv_act:   ", dv_act.shape)

    f1 = plot_traj(traj_dict={"old": trajs_np[0], "new": trajs_np[1]}, plot_cols={'x': 0, 'y': 1, 'yaw': 2, 'p': 3, 'q': 4, 'r': 5}, tau=100, fig=True)
    f2 = plot_traj(traj_dict={"old": dv_act[0], "new": dv_act[1]}, plot_cols={'du': 0, 'dv': 1, 'dr': 2, 'Fx': 3, 'Fy': 4, 'Tz': 5}, tau=100, fig=True)
    print()
    plt.show()
    #plt.close('all')

    dv_preds = [FakeDv(traj_dv[None]).to(device) for traj_dv in trajs_dv]
    step = [AUVRNN2Dstep().to(device) for dv_pred in dv_preds]
    for s, d in zip(step, dv_preds):
        s.dv_pred = d

    preds = [AUVRNN2D().to(device) for s in step]
    for p, s in zip(preds, step):
        p.step = s

    '''
    ts = [pred(traj[None, 0:1], act[None, 0:tau]) for pred, traj, act in zip(preds, trajs, acts)]
    
    gt_ts = [traj[0:tau] for traj in trajs]
    
    gt_ts_dv = [traj_dv[1:tau+1] for traj_dv in trajs_dv]

    print("t:        ", [t.shape for t in ts])
    print("gt_t:     ", [gt_t.shape for gt_t in gt_ts])

    print("t_dv:     ", t_dv.shape)
    print("gt_t_dv:  ", gt_t_dv.shape)

    print("dv diff:  ", (t_dv - gt_t_dv).sum())

    print("x diff:   ", (torch.abs(t[:, 0] - gt_t[:, 0])).sum())
    print("u diff:   ", (torch.abs(t[:, 3] - gt_t[:, 3])).sum())

    print("y diff:   ", (torch.abs(t[:, 1] - gt_t[:, 1])).sum())
    print("v diff:   ", (torch.abs(t[:, 4] - gt_t[:, 4])).sum())

    print("yaw diff: ", (torch.abs(t[:, 2] - gt_t[:, 2])).sum())
    print("r diff:   ", (torch.abs(t[:, 5] - gt_t[:, 5])).sum())

    t_dict = {"gt": gt_t, "t": t}
    img = gen_imgs(t_dict, [tau])[0]

    plt.imshow(img)
    plt.show()
    '''


def assesOPS():
    dt = 0.1
    pose = torch.Tensor(np.array([[0.1, 0.5, np.pi/4.]]))
    vel = torch.Tensor(np.array([[0.1, 0.1, np.pi/10.]]))
    dv = torch.Tensor(np.array([[0.01, 0.01, np.pi/100.]]))

    toSE2 = ToSE2Mat()
    intRight = SE2Int()
    invSE2 = SE2Inv()
    adjSE2 = SE2Adj()
    flatSE2 = FlattenSE2()

    X = toSE2(pose)
    print("\nX:\n", X)

    incr = vel*dt

    X_next = intRight(X, incr)
    print("\nX_next:\n", X_next)

    X_next_inv = invSE2(X_next)
    print("\nX_next_inv:\n", X_next_inv)

    adj_X = adjSE2(X)
    print("\nAdj_X:\n", adj_X)

    v_i = (adj_X @ vel[..., None])[..., 0]
    print("\nVel inertial\n", v_i)

    v_i_next = v_i + dv
    print("\nVel inertial next\n", v_i_next)

    adj_X_next_inv = adjSE2(X_next_inv)
    print("\nAdj x next inv\n", adj_X_next_inv)

    v_next = (adj_X_next_inv @ v_i_next[..., None])[..., 0]
    print("\nv_next\n", v_next)

    s_next = flatSE2(X_next, v_next)
    print("\nS_next\n", s_next)


if __name__ == "__main__":
    inspect_bag()

