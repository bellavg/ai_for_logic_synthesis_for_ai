import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from VAE import VAE

if __name__ == '__main__':
    # Load MNIST dataset
    train = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=2)

    # Define VAE model parameters
    input_dim, hidden_dims = 16, [32, 128]
    latent_size = 2

    # Initialize the VAE model
    vae_model = VAE(input_dim, latent_size, hidden_dims)

    # Separate encoder and decoder
    encoder = vae_model.encoder
    decoder = vae_model.decoder

    # Retrieve all parameters of both models
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    opt = Adam(lr=0.003, params=parameters)

    # Training loop
    for epoch in range(5):
        for images, _ in tqdm(trainloader):  # if tqdm gives you trouble just remove it
            b, c, h, w = images.size()

            # Forward pass
            o, kl = vae_model(images)

            # Reconstruction loss
            rec = F.binary_cross_entropy(o, images, reduction='none')
            rec = rec.view(b, c * h * w).sum(dim=1)

            # Sum the losses and take the mean
            loss = (rec + kl).mean()
            loss.backward()

            opt.step()
            opt.zero_grad()

        print(f'Epoch {epoch}: Loss = {loss.item()}')

    # Save the trained model
    torch.save(vae_model.state_dict(), 'vae_model.pth')
