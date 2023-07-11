
import onnxruntime
import numpy as np

def load_onnx_model(onnx_model_filename, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
    opts = onnxruntime.SessionOptions()

    # opts.intra_op_num_threads = 2
    # opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    # opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    # opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # opts.enable_profiling = True
    # opts.log_severity_level = 0
    # opts.log_verbosity_level= 0

    return onnxruntime.InferenceSession(onnx_model_filename, opts, providers)

def create_onnx_bound_model(onnx_model_filename, state, seq, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
    opts = onnxruntime.SessionOptions()
    bind_session = onnxruntime.InferenceSession(onnx_model_filename, opts, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    io_binding = bind_session.io_binding()

    io_binding.bind_input(
        name=bind_session.get_inputs()[0].name,
        device_type='cuda',
        device_id=0,
        element_type=np.float32,
        buffer_ptr=state.data_ptr(),
        shape=state.shape)

    io_binding.bind_input(
        name=bind_session.get_inputs()[1].name,
        device_type='cuda',
        device_id=0,
        element_type=np.float32,
        buffer_ptr=seq.data_ptr(),
        shape=seq.shape)

    io_binding.bind_output(
        name=bind_session.get_outputs()[0].name,
        device_type='cuda',
        device_id=0,)

    return bind_session, io_binding

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()