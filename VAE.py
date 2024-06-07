import torch
import torch.nn as nn

def kl_loss(zmean, zsig):
    b, l = zmean.size()

    kl = 0.5 * torch.sum(zsig.exp() - zsig + zmean.pow(2) - 1, dim=1)
    # -- The KL divergence between a given normal distribution and a standard normal distribution
    #    can be rewritten this way. It's a good exercise to work this out.

    assert kl.size() == (b,)
    # -- At this point we want the loss to be a single value of each instance in the batch.
    #    Asserts like this are a good way to document what you know about the shape of the
    #    tensors you deal with.

    return kl

def sample(zmean, zsig):
    b, l = zmean.size()

    # sample epsilon from a standard normal distribution
    eps = torch.randn(b, l)

    # transform eps to a sample from the given distribution
    return zmean + eps * (zsig * 0.5).exp()


# Pytorch doesn't give us a Reshape module. We'll add that ourselves so we
# can define the encoder and decoder as sequences of operations OLD NEWS BUT NOT GOING TO BOTHER TO CHANGE
class Reshape(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view((input.size(0),) + self.shape)  # keep the batch dimensions, reshape the rest

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, alpha=1.0, beta=1.0):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim1 = hidden_dims[0]
        self.hidden_dim2 = hidden_dims[1]
        self.alpha = alpha
        self.beta = beta
        self.encoder = nn.Sequential(
    nn.Conv2d(1, self.input_dim, (3, 3), padding=1), nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(self.input_dim, self.hidden_dim1, (3, 3), padding=1), nn.ReLU(),
    nn.Conv2d(self.hidden_dim1, self.hidden_dim1, (3, 3), padding=1), nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Conv2d(self.hidden_dim1, self.hidden_dim2, (3, 3), padding=1), nn.ReLU(),
    nn.Conv2d(self.hidden_dim2, self.hidden_dim2, (3, 3), padding=1), nn.ReLU(),
    nn.MaxPool2d((2, 2)),
    nn.Flatten(),
    nn.Linear(3 * 3 * self.hidden_dim2, 2 * self.latent_dim)
)
        self.decoder = nn.Sequential(
    nn.Linear(self.latent_dim, self.hidden_dim2 * 3 * 3), nn.ReLU(),
    Reshape((self.hidden_dim2, 3, 3)),
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.ConvTranspose2d(self.hidden_dim2, self.hidden_dim2, (3, 3), padding=1), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.ConvTranspose2d(self.hidden_dim2, self.hidden_dim1, (3, 3), padding=0), nn.ReLU(), # note the padding
    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
    nn.ConvTranspose2d(self.hidden_dim1, 1, (3, 3), padding=1), nn.Sigmoid()
)

    def reparametrize(self, z):
        # - split z into mean and sigma
        zmean, zsig = z[:, :self.latent_dim], z[:, self.latent_dim:]
        kl = kl_loss(zmean, zsig)

        zsample = sample(zmean, zsig)
        return zsample, kl

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)

        zsample, kl = self.reparametrize(z)


        o = self.decode(zsample)

        return o, kl
