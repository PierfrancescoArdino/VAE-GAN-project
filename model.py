import pytorch_lightning as pl
from torch import nn
import torch
from torch.autograd import Variable
import numpy as np
from losses import losses
from collections import OrderedDict
import torchvision
from torchsummary import summary
from base_model import BaseModel
import os

def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class Encoder(nn.Module):

    def __init__(self, ngf=32, z_dim=512):
        super(Encoder, self).__init__()

        self.f_dim = ngf
        self.z_dim = z_dim
        self.input_dim = 3

        self.conv0 = nn.Sequential(
            nn.Conv2d(self.input_dim, self.f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.ELU(),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.f_dim, self.f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.f_dim * 2, self.f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 4, affine=False),
            nn.ELU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.f_dim * 4, self.f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 8, affine=False),
            nn.ELU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(self.f_dim * 8, self.f_dim * 16,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 8, affine=False),
            nn.ELU(),
        )
        self.fc_mu = nn.Sequential(
            nn.Conv2d(self.f_dim * 16, self.z_dim, kernel_size=4, stride=1, padding=0)
        )
        self.fc_logvar = nn.Sequential(
            nn.Conv2d(self.f_dim * 16, self.z_dim, kernel_size=4, stride=1, padding=0)
        )

    def forward(self, img):
        e0 = self.conv0(img)
        e1 = self.conv1(e0)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        fc_mu = self.fc_mu(e4).squeeze(2).squeeze(2)
        logvar = self.fc_logvar(e4).squeeze(2).squeeze(2)

        return fc_mu, logvar


class Decoder(nn.Module):

    def __init__(self, ngf=32, z_dim=512):
        super(Decoder, self).__init__()
        self.z_dim = z_dim
        self.f_dim = ngf
        self.conv0 = nn.Sequential(
            nn.ConvTranspose2d(self.z_dim, self.f_dim * 16, kernel_size=4, stride=1, padding=0),
            nn.InstanceNorm2d(self.f_dim * 16, affine=False),
            nn.ELU(),
        )
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim * 16, self.f_dim * 8,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 8, affine=False),
            nn.ELU(),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim * 8, self.f_dim * 4,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 4, affine=False),
            nn.ELU(),
        )
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim * 4, self.f_dim * 2,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim * 2, affine=False),
            nn.ELU(),
        )
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim * 2, self.f_dim,
                      kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(self.f_dim, affine=False),
            nn.ELU(),
        )
        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(self.f_dim, 3,
                      kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):

        dec0 = self.conv0(z.unsqueeze(2).unsqueeze(2))
        dec1 = self.conv1(dec0)
        dec2 = self.conv2(dec1)
        dec3 = self.conv3(dec2)
        dec4 = self.conv4(dec3)
        instance = self.conv5(dec4)


        return instance


class Discriminator(nn.Module):

    def __init__(self, input_nc=6, ndf=32, n_layers=3, norm_layer=nn.InstanceNorm2d,
                 use_sigmoid=False, num_D=1, getIntermFeat=False,
                 use_sn_discriminator=False):
        super(Discriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
        self.use_sn_discriminator = use_sn_discriminator
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer,
                                       use_sigmoid, getIntermFeat,
                                       self.use_sn_discriminator)
            if getIntermFeat:

                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j),
                            getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1],
                                       count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            shapes = [input.shape]
            for i in range(len(model)):
                result.append(model[i](result[-1]))

            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        shapes = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self,
                                 'scale' + str(num_D - 1 - i) + '_layer' + str(
                                     j)) for j in range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            res = self.singleD_forward(model, input_downsampled)
            result.append(res)
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False,
                 use_sn_discriminator=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        self.use_sn_discriminator = use_sn_discriminator
        kw = 4
        padw = int(np.floor((kw-1.0)/2))
        sequence = [
            [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
             nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)
        print("Done!")
    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)


class VaeGanModule(BaseModel):


    def init_loss_filter(self):
        flags = (True, True, True, True, True)
        def loss_filter(g_gan, g_kl_image, g_image_rec, d_real, d_fake):
                return [l for (l, f) in
                        zip((g_gan, g_kl_image, g_image_rec, d_real, d_fake), flags) if f]
        return loss_filter


    def __init__(self, opt, device):
        super().__init__(opt)
        self.ngf = opt.ngf
        self.z_dim = opt.z_dim
        self.hparams = opt
        self.encoder = Encoder(ngf=self.ngf, z_dim=self.z_dim).to(device)
        self.encoder.apply(weights_init)
        self.decoder = Decoder(ngf=self.ngf, z_dim=self.z_dim).to(device)
        self.decoder.apply(weights_init)
        self.discriminator = Discriminator().to(device)
        self.discriminator.apply(weights_init)
        self.criterionFeat = torch.nn.L1Loss()
        self.criterionGAN = losses.GANLoss(gan_mode="lsgan")
        self.last_imgs = None
        self.save_dir = os.path.join(opt.checkpoints_dir, "DPFIP", opt.name)
        self.loss_filter = self.init_loss_filter()
        params = list(self.encoder.parameters()) + \
                                  list(self.decoder.parameters())
        self.optimizer_vae = torch.optim.Adam(params, lr=opt.lr,
                                            betas=(opt.beta1, opt.beta2))
        # optimizer D
        params = list(self.discriminator.parameters())

        self.optimizer_D = torch.optim.Adam(params, lr=opt.lr,
                                            betas=(opt.beta1, opt.beta2))
        self.loss_names = \
            self.loss_filter('G_GAN', "G_KL_image",
                             "G_Image_Rec",
                             'D_real', 'D_fake')

    def reparameterize(self, mu, logvar, mode):
        if mode == 'train':
            std = torch.exp(0.5 * logvar)
            eps = Variable(std.data.new(std.size()).normal_())
            return mu + eps * std
        else:
            return mu

    def discriminate(self, fake_image, real_image):
        input_concat_fake = \
            torch.cat((fake_image.detach(), real_image), dim=1) # non sono sicuro che .detach() sia necessario in lightning
        input_concat_real = \
            torch.cat((real_image, real_image),
                      dim=1)
        return self.discriminator.forward(input_concat_fake), \
               self.discriminator.forward(input_concat_real)

    def forward(self, images):
        x = images
        self.last_imgs = x

        mu, log_var = self.encoder(x)
        z_repar = self.reparameterize(mu, log_var, mode="train")
        # decode
        fake_image = self.decoder(z_repar)
        # how well can it label as real?
        pred_fake, pred_real = self.discriminate(fake_image, x)
        loss_D_fake = self.criterionGAN.forward(pred_fake, False)

        # Real Loss

        loss_D_real = self.criterionGAN(pred_real, True)

        # reconstruction
        reconstruction_loss = self.criterionFeat(fake_image, x)
        kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        input_concat_fake = \
            torch.cat((fake_image, x), dim=1)
        pred_fake = self.discriminator.forward(input_concat_fake)
        loss_G_GAN = self.criterionGAN(pred_fake, True)
        
        return [self.loss_filter(loss_G_GAN, kld_loss, reconstruction_loss, loss_D_real, loss_D_fake), fake_image]

    def save(self, which_epoch):
        self.save_network(self.encoder, 'encoder', which_epoch, self.gpu_ids)
        self.save_network(self.decoder, 'decoder', which_epoch, self.gpu_ids)
        self.save_network(self.discriminator, 'discriminator', which_epoch,
                          self.gpu_ids)

