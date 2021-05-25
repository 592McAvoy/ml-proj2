import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel


class EncBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncBlock, self).__init__()
        self.components = nn.Sequential(
            nn.Conv2d(in_c, out_c,
                      kernel_size=3, stride=2, padding=1),  # shrink size to half
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU())

    def forward(self, x):
        return self.components(x)


class DecBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(EncBlock, self).__init__()
        self.components = nn.Sequential(
            nn.ConvTranspose2d(in_c,  out_c,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU())

    def forward(self, x):
        return self.components(x)


class VanillaVAE(BaseModel):

    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_base,
                 hidden_layer,
                 **kwargs):
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        # Build Encoder
        enc_modules = []
        enc_modules.append(
            EncBlock(in_c=in_channels, out_c=hidden_base)
        )

        for i in range(1, hidden_layer):
            mul = 2**i
            enc_modules.append(
                EncBlock(in_c=hidden_base*mul, out_c=hidden_base*mul*2)
            )

        self.encoder = nn.Sequential(*enc_modules)

        final_hidden = hidden_base*(2**hidden_layer)//2
        self.fc_mu = nn.Linear(final_hidden*4, latent_dim)
        self.fc_var = nn.Linear(final_hidden*4, latent_dim)

        # Build Decoder    

        self.decoder_input = nn.Linear(latent_dim, final_hidden * 4)

        dec_modules = []
        for i in range(hidden_layer-1, 0, -1):
            mul = 2**i
            modules.append(
                DecBlock(in_c=hidden_base*mul, out_c=hidden_base*mul//2)
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh())

    def encode(self, input: Tensor):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs):
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 +
                              log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': -kld_loss}

    def sample(self,
               num_samples,
               current_device, **kwargs):
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs):
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
