import torch
from utils.models.SalMM import SalMM

model = SalMM()
input_dims = (torch.randn(1, 3, 256, 256),)

torch.onnx.export(
    model,
    input_dims,
    "test.onnx",
    opset_version=15,
    input_names=["input"],
    output_names=["output"],
    
)
