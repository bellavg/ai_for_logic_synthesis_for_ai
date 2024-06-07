from matplotlib import pyplot as plt
import numpy as np
import torch
import onnx
from mlp import MLP
# Visualize the ONNX model with Netron
import netron

#netron.start("mlp_model.onnx")

# Load and inspect the ONNX model
model = onnx.load("mlp_model.onnx")
weights_dict = {}
for tensor in model.graph.initializer:
    weights_dict[tensor.name] = np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(tensor.dims)
    print(f"Layer: {tensor.name}")
    print(f"Shape: {tensor.dims}")
    print(f"Weights: {np.frombuffer(tensor.raw_data, dtype=np.float32)[:5]}...")  # Print the first 5 weights for brevity
    print()

# Example: Visualize weights of the first layer (fc1)
fc1_weights = weights_dict['fc1.weight']

fig, axes = plt.subplots(4, 8, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    if i < fc1_weights.shape[0]:
        ax.imshow(fc1_weights[i].reshape(28, 28), cmap='plasma')
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('MLP Layer 1 Weights from ONNX', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('mlp_onnx_weights_fc1.png')
plt.show()