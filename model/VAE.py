import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from model.component.VGG_loss import VGGPerceptualLoss


class EncBlock(nn.Module):
    def __init__(self, in_c, out_c, norm_func=nn.BatchNorm2d):
        super(EncBlock, self).__init__()
        self.components = nn.Sequential(
            nn.Conv2d(in_c, out_c,
                      kernel_size=3, stride=2, padding=1),  # shrink size to half
            norm_func(out_c),
            nn.LeakyReLU())

    def forward(self, x):
        return self.components(x)


class DecBlock(nn.Module):
    def __init__(self, in_c, out_c, norm_func=nn.BatchNorm2d):
        super(DecBlock, self).__init__()
        self.components = nn.Sequential(
            nn.ConvTranspose2d(in_c,  out_c,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            norm_func(out_c),
            nn.LeakyReLU())

    def forward(self, x):
        return self.components(x)


class VAE(BaseModel):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 hidden_base,
                 recons_type='mse',
                 norm='bn',
                 hidden_layer=5):
        super().__init__()

        
        self.recons_type = recons_type
        if 'vgg' in recons_type:
            self.vgg = VGGPerceptualLoss().requires_grad_(False)

        self.latent_dim = latent_dim
        norm_func = nn.InstanceNorm2d if norm=='in' else nn.BatchNorm2d

        # Build Encoder
        enc_modules = []
        enc_modules.append(
            EncBlock(in_c=in_channels, out_c=hidden_base)
        )

        for i in range(1, hidden_layer):
            mul = 2**i
            # if i==hidden_layer-1:
            #     enc_modules.append(
            #         EncBlock(in_c=hidden_base*mul//2, out_c=hidden_base*mul, norm_func=nn.BatchNorm2d)
            #     )
            #     continue
            enc_modules.append(
                EncBlock(in_c=hidden_base*mul//2, out_c=hidden_base*mul)
            )

        self.encoder = nn.Sequential(*enc_modules)

        final_hidden = hidden_base*(2**hidden_layer)//2
        self.final_hidden = final_hidden
        self.fc_mu = nn.Linear(final_hidden*4, latent_dim)
        self.fc_var = nn.Linear(final_hidden*4, latent_dim)

        # Build Decoder
        self.decoder_input = nn.Linear(latent_dim, final_hidden * 4)

        dec_modules = []
        for i in range(hidden_layer-1, 0, -1):
            mul = 2**i            
            dec_modules.append(
                DecBlock(in_c=hidden_base*mul, out_c=hidden_base*mul//2, norm_func=norm_func)
            )

        self.decoder = nn.Sequential(*dec_modules)

        self.final_layer = nn.Sequential(
            DecBlock(in_c=hidden_base, out_c=hidden_base, norm_func=norm_func),
            nn.Conv2d(hidden_base, out_channels=in_channels,
                      kernel_size=3, padding=1),
            nn.Tanh())

        self._weight_init()
    

    def encode(self, x):
        """
        img (b,c,h,w) -> mu & var
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)

        return mu, log_var

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        z ~ N(mu, var)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, self.final_hidden, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return out, mu, log_var

    def recons_loss(self, x, y):
        if self.recons_type == 'mse':
            recons_loss = F.mse_loss(x, y)
        elif self.recons_type == 'l1':
            recons_loss = F.l1_loss(x, y)
        elif self.recons_type == 'vgg':
            recons_loss = self.vgg(x, y, feature_layers=[0, 1])
        else:
            raise NotImplementedError(self.recons_type)
        # recons_loss = self.vgg(x, y, )+F.l1_loss(x, y)
        return recons_loss

    def kld_loss(self, mu, logvar):
        # KL(N(\mu, \sigma), N(0, 1))
        # = 1/2 \sum (1 + \log \sigma^2 - \sigma^2 - \mu^2}
        kld_loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
        # kld_loss = torch.mean(torch.sum(kld_loss, dim=1), dim=0) # sum in dim, mean in batch
        kld_loss = torch.mean(kld_loss)
        
        return kld_loss

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)




if __name__ == '__main__':
    model = VAE(in_channels=3, latent_dim=128, hidden_base=32, hidden_layer=5)
    print(model)

    b,c,h,w = 10, 3, 64, 64
    dummy = torch.rand((b,c,h,w))
    # mu, var = model.encode(dummy)
    out, mu, var = model(dummy)
    print(out.size(), mu.size(), var.size())
