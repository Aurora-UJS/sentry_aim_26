import onnxruntime as ort

session = ort.InferenceSession("opt-0625-001.onnx")
inputs = session.get_inputs()

for i in inputs:
    print("输入名:", i.name)
    print("形状:", i.shape)
    print("类型:", i.type)
import onnx
model = onnx.load("opt-0625-001.onnx")
for output in model.graph.output:
    print(f"输出名: {output.name}")
    print(f"类型: {output.type.tensor_type.elem_type}")
    dims = output.type.tensor_type.shape.dim
    shape = [d.dim_value for d in dims]
    print(f"形状: {shape}")
