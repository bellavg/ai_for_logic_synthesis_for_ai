import torch
import onnx
import numpy as np
from mlp import MLP


model = MLP()
# Convert to ONNX
model.load_state_dict(torch.load('best_mlp_mnist.pth'))
model.eval()
dummy_input = torch.randn(1, 1, 28, 28)  # Adjust dimensions to your input shape
torch.onnx.export(model, dummy_input, "mlp_model.onnx", verbose=True)

# Load and inspect the ONNX model
model = onnx.load("mlp_model.onnx")
for tensor in model.graph.initializer:
    print(f"Layer: {tensor.name}")
    print(f"Shape: {tensor.dims}")
    print(f"Weights: {np.frombuffer(tensor.raw_data, dtype=np.float32)}")
    print()
