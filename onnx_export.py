import os
import torch
from utils.models.SalMM import SalMM

os.environ['EXPORTING_ONNX_MODEL'] = "1"

model = SalMM()
input_example = (torch.randn(1, 3, 256, 256),)

torch.onnx.export(
    model,
    input_example,
    "test.onnx",
    opset_version=15,
    input_names=["input"],
    output_names=["output"],
)
