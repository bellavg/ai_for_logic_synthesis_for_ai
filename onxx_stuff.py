import torch
import onnx
import numpy as np
from VAE import VAE


vae_model = VAE(16, 2, [32, 128])
# Convert to ONNX
vae_model.load_state_dict(torch.load('vae_model.pth'))
vae_model.eval()
dummy_input = torch.randn(1, 1, 28, 28)  # Adjust dimensions to your input shape
torch.onnx.export(vae_model, dummy_input, "vae_model.onnx", verbose=True)

# Load and inspect the ONNX model
model = onnx.load("vae_model.onnx")
for tensor in model.graph.initializer:
    print(f"Layer: {tensor.name}")
    print(f"Shape: {tensor.dims}")
    print(f"Weights: {np.frombuffer(tensor.raw_data, dtype=np.float32)}")
    print()
