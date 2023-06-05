import torch
import warnings
from nn_utile import AUVTraj
import torch.onnx
import onnxruntime


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

# Export the model
torch.onnx.export(model,
        args=(state, seq),
        f="model.onnx",
        export_params=True,
        opset_version=15, # Check different opsets. Must be >=11
        # opset_version=11,
        # verbose=True, 
        # training=False,
        # do_constant_folding=True,
        input_names=["input_1", "input_2"],
        output_names=["output_1","output_2","output_3"])
