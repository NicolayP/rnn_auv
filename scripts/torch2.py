import torch
import pypose as pp
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
import warnings
from nn_utile import AUVRNNDeltaV, AUVTraj

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
        if not use_cuda:
            warnings.warn("Asked for GPU but torch couldn't find a Cuda capable device")
    return torch.device(f"cuda:{unit}" if use_cuda else "cpu")

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    restul = fn()
    end.record()
    torch.cuda.synchronize()
    return restul, start.elapsed_time(end) / 1000

def gen_data(b, device):
    state = torch.zeros(size=(b, 1, 13), device=device)
    state[..., 6] = 1.
    seq = torch.zeros(size=(b, 50, 6), device=device)
    return state, seq

def evaluate(mod, state, seq):
    return mod(state, seq)

evaluate_opt = torch.compile(evaluate, fullgraph=True)

N_ITERS = 10
BATCH_SIZE = 2000

device = get_device(True)
state, seq = gen_data(BATCH_SIZE, device)

# Warm-up
model = AUVTraj().to(device)

print("eager:", timed(lambda: evaluate(model, state, seq))[1])
print("compile:", timed(lambda: evaluate_opt(model, state, seq))[1])



eager_times = []
compile_times = []
for i in range(N_ITERS):
    _, eager_time = timed(lambda: evaluate(model, state, seq))
    eager_times.append(eager_time)
    print(f"eager eval time {i}: {eager_time}")

print("~" * 10)

compile_times = []
for i in range(N_ITERS):
    _, compile_time = timed(lambda: evaluate_opt(model, state, seq))
    compile_times.append(compile_time)
    print(f"compile eval time {i}: {compile_time}")
print("~" * 10)


eager_med = np.median(eager_times)
compile_med = np.median(compile_times)
speedup = eager_med / compile_med
print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
print("~" * 10)