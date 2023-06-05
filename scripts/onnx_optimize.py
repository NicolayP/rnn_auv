import onnxruntime
from onnx import load_model, save_model
# from onnxconverter_common import float16
from onnxoptimizer import optimize
from onnxsim import simplify

init_file = "./model.onnx"
opt_file = "./model_optimized.onnx"

model_onnx = load_model(init_file)
print("Loaded model")

print("Converting Float to Float16")
# Optional transformers alternative
# from onnxruntime.transformers.float16 import convert_float_to_float16
# optimized_model = convert_float_to_float16(model_onnx, keep_io_types=True, force_fp16_initializers=False, disable_shape_infer=False) # TODO: test with initializers=false
# optimized_model = float16.convert_float_to_float16(model_onnx)

# TODO: This seems to break the model a bit
print("Simplifying the model")
optimized_model, _ = simplify(model_onnx)
save_model(optimized_model, "./model_simplified.onnx", save_as_external_data=True, all_tensors_to_one_file=True)

print("Removing unnecessary initializers")
passes = ["extract_constant_to_initializer", "eliminate_unused_initializer"]
optimized_model = optimize(optimized_model, passes)

save_model(optimized_model, opt_file, save_as_external_data=True, all_tensors_to_one_file=True)
print("Saved optimized model")
