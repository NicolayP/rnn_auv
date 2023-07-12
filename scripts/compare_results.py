import torch
import warnings
from nn_utile import AUVTraj, AUVRNNDeltaV
import torch.onnx
import onnxruntime
import pypose as pp


from network_utils import load_onnx_model, to_numpy

from utile import to_euler
from utile import plot_traj
import matplotlib.pyplot as plt


gpu_ok = False
if torch.cuda.is_available():
    device_cap = torch.cuda.get_device_capability()
    if device_cap in ((7, 0), (8, 0), (9, 0)):
        gpu_ok = True

if not gpu_ok:
    warnings.warn(
        "GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower "
        "than expected."
    )

def get_device(gpu=False, unit=0):
    use_cuda = False
    if gpu:
        use_cuda = torch.cuda.is_available()
        print(torch.cuda.is_available())
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")
    return torch.device(f"cuda:{unit}" if use_cuda else "cpu")

def gen_data(b, device):
    state = torch.zeros(size=(b, 1, 13), device=device)
    state[..., 6] = 1.
    seq = torch.zeros(size=(b, 50, 6), device=device)
    return state, seq


N_ITERS = 10
BATCH_SIZE = 2000

device = get_device(True)
state, seq = gen_data(BATCH_SIZE, device)



# Warm-up
model = AUVTraj().to(device)

scripted_model = torch.jit.script(model, (state, seq))
onnx_model_filename = "../model.onnx"
ort_session = load_onnx_model(onnx_model_filename, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
ort_inputs = {
    ort_session.get_inputs()[0].name: to_numpy(state),
    ort_session.get_inputs()[1].name: to_numpy(seq),
}


model_trajs, model_vels, model_dvs = model(state, seq)
scripted_trajs, scripted_vels, scripted_dvs = scripted_model(state, seq)
onnx_trajs, onnx_vels, onnx_dvs = ort_session.run(None, ort_inputs)

model_trajs = model_trajs.detach().cpu()
scripted_trajs = scripted_trajs.detach().cpu()

tau = 20

model_traj_euler = to_euler(model_trajs[0].data)
scripted_traj_euler = to_euler(scripted_trajs[0].data)
onnx_traj_euler = to_euler(onnx_trajs[0])
s_col = {"x": 0, "y": 1, "z": 2, "roll": 3, "pitch": 4, "yaw": 5}
plot_traj({"eager": model_traj_euler, "compiled": scripted_traj_euler, "onnx": onnx_traj_euler}, s_col, tau, True, title="State")
plt.show()



