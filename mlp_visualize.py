import torch
from torchviz import make_dot
from mlp import MLP  # Ensure your MLP model is defined in model.py

import matplotlib.pyplot as plt
import numpy as np

model = MLP()
# Load trained model if available
model.load_state_dict(torch.load('best_mlp_mnist.pth'))


weights = model.fc1.weight.data.numpy()

fig, axes = plt.subplots(4, 8, figsize=(12, 6))

# Adjusting font sizes and color map, and removing axis labels
for i, ax in enumerate(axes.flat):
    if i < weights.shape[0]:
        ax.imshow(weights[i].reshape(28, 28), cmap='plasma')
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle('MLP Layer 1 Weights', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('mlp_weights.png')
plt.show()


model.eval()

# Visualize the model architecture
x = torch.randn(1, 1, 28, 28)  # Dummy input
y = model(x)
make_dot(y, params=dict(model.named_parameters())).render("mlp_architecture", format="png")