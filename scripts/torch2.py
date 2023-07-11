import torch
import numpy as np
import warnings
from nn_utile import AUVRNNDeltaV, AUVTraj, AUVStep

# from nn_utile import SE3Type
import torch.onnx

from network_utils import load_onnx_model, to_numpy

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
    state[..., 6] = 1.0
    seq = torch.zeros(size=(b, 50, 6), device=device)
    return state, seq


N_ITERS = 30
BATCH_SIZE = 2000

device_id = 0

device = get_device(True, device_id)
state, seq = gen_data(BATCH_SIZE, device)


def run_model(model_func, iters, tag):
    times = []
    for i in range(iters):
        _, time = timed(model_func)
        times.append(time)
        print(f"{tag} eval time {i}: {time}")
    return times


import torch
import click


@click.command()
@click.option("--script_model", type=str, help="Operation to perform: save, run, all")
@click.option("--onnx_model", type=str, help="Operation to perform: save, run, all")
def main(script_model, onnx_model):
    # Warm-up
    model = AUVTraj().to(device)

    if script_model:
        compiled = torch.jit.load(script_model)
    else:
        compiled = torch.jit.script(model, (state, seq))
        compiled.save("scripted_model.pt")

    if not onnx_model:
        onnx_model = "model.onnx"
        torch.onnx.export(
            model,
            args=(state, seq),
            f=onnx_model,
            export_params=True,
            opset_version=15,
            # opset_version=11,
            # verbose=True,
            # training=False,
            # do_constant_folding=True,
            input_names=["input_1", "input_2"],
            output_names=["output_1", "output_2", "output_3"],
        )

    ort_session = load_onnx_model(onnx_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    ort_inputs = {
        ort_session.get_inputs()[0].name: to_numpy(state),
        ort_session.get_inputs()[1].name: to_numpy(seq),
    }

    print("\n" + "~" * 10)
    N_ITERS = 10
    eager_times = run_model(lambda: model(state, seq), N_ITERS, "eager")
    print("~" * 10)
    compile_times = run_model(lambda: compiled(state, seq), N_ITERS, "compiled")
    print("~" * 10)
    onnx_times = run_model(lambda: ort_session.run(None, ort_inputs), N_ITERS, "onnx")
    print("~" * 10)

    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    onnx_med = np.median(onnx_times)
    print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {eager_med/compile_med}x")
    print(f"(eval) eager median: {eager_med}, onnx median: {onnx_med}, speedup: {eager_med/onnx_med}x")


if __name__ == "__main__":
    main()
