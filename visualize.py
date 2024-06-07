import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import numpy as np
import torch
import torchvision
from VAE import VAE


train = torchvision.datasets.MNIST(root='./mnist', train=True, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True, num_workers=2)

# gather up first 200 batches into one big tensor
numbatches = 200  # -- change 200 to a lower number to speed things up
images, labels = [], []
for i, (ims, lbs) in enumerate(trainloader):
    images.append(ims)
    labels.append(lbs)

    if i > numbatches:
        break

images, labels = torch.cat(images, dim=0), torch.cat(labels, dim=0)

n, c, h, w = images.size()

model = VAE(16, 2, [32, 128])
encoder = model.encoder
z = encoder(images)
latents = z[:, :2].data

mn, mx = latents.min(), latents.max()
size = 1.0 * (mx - mn) / math.sqrt(n)
# Change 0.75 to any value between ~ 0.5 and 1.5 to make the digits smaller or bigger

fig = plt.figure(figsize=(16, 16))

# colormap for the images
norm = mpl.colors.Normalize(vmin=0, vmax=9)
cmap = mpl.cm.get_cmap('tab10')

for i in range(n):
    x, y = latents[i, 0:2]
    l = labels[i]

    im = images[i, :]
    alpha_im = im.permute(1, 2, 0).numpy()
    color = cmap(norm(l))
    color_im = np.asarray(color)[None, None, :3]
    color_im = np.broadcast_to(color_im, (h, w, 3))
    # -- To make the digits transparent we make them solid color images and use the
    #    actual data as an alpha channel.
    #    color_im: 3-channel color image, with solid color corresponding to class
    #    alpha_im: 1-channel grayscale image corresponding to input data

    im = np.concatenate([color_im, alpha_im], axis=2)
    plt.imshow(im, extent=(x, x + size, y, y + size))

    plt.xlim(mn, mx)
    plt.ylim(mn, mx)
    plt.savefig('latent_space.png')
    plt.show()
