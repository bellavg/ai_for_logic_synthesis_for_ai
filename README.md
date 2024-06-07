# ai_for_logic_synthesis_for_ai

need torch, torchvision, tqdm, onnx, numpy, matplotlib

Notes on model from Peter Bloem: We are cheating a little with the output activation. To truly follow the derivation of the VAE, this should define a distribution in the data space (which are continuous numbers _between_ 0 and 1). In this case, we use a sigmoid activation with a binary cross-entropy, which would be a distribution for binary data (either 0 or 1). 

A more correct VAE would use, for instance, a Gaussian output, which boils down to an MSE loss (but doesn't work well for this task), or add some terms to make the BCE loss work theoretically. For now we'll ignore this and stick with a plain BCE loss.

More importantly, note that the output of the encoder is _twice_ the size of the latent space, while the input to the decoder is the size of the latent space. This is because the decoder gives us a mean _and a variance_ on the latent space, from which we'll _sample_ the input to the decoder.

We'll also need to compute the KL loss term from this mean and variance. We'll introduce some utility functions for both operations.