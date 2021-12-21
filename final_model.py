import torch
import torch.nn as nn
from torch.autograd import Variable

import torch
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.current_device()

# adapted from pytorch/examples/vae and ethanluoyc/pytorch-vae
 
class ImageClassifier(nn.Module):
    def __init__(self, latent_variable_size, pretrained, nout=2):
        super(ImageClassifier, self).__init__()
        self.latent_variable_size = latent_variable_size
        self.feature_extractor = pretrained
        self.classifier = nn.Linear(latent_variable_size, nout)

    def forward(self, x):
        x, _ = self.feature_extractor.encode(x)
        x = self.classifier(x.view(-1, self.latent_variable_size))
        return x

class VAE(nn.Module):
    def __init__(self, nc=1, ngf=128, ndf=128, latent_variable_size=128, imsize=64, batchnorm=False):
        super(VAE, self).__init__()
 
        self.nc = nc #number_channels
        self.ngf = ngf #number of filters in generator
        self.ndf = ndf #number of filters in discriminator
        self.imsize = imsize
        self.latent_variable_size = latent_variable_size
        self.batchnorm = batchnorm
 
        self.encoder = nn.Sequential(
            # input is 3 x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), #in_channels=1, out_channels=128, kernel_size=4, stride=2, padding=1
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 2 x 2
        )
 
        self.fc1 = nn.Linear(ndf*8*2*2, latent_variable_size) #self.z-mean (4096=64*64)
        self.fc2 = nn.Linear(ndf*8*2*2, latent_variable_size) #self.z_log_var (4096=64*64)
 
        # decoder
 
        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            # state size. (ngf*8) x 2 x 2
            nn.ConvTranspose2d(ngf * 8, ngf * 8, 4, 2, 1, bias=False), #in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Sigmoid(),
            # state size. (nc) x 64 x 64
        )
 
        self.d1 = nn.Sequential(
            nn.Linear(latent_variable_size, ngf*8*2*2),
            nn.ReLU(inplace=True),
            )

        self.bn_mean = nn.BatchNorm1d(latent_variable_size)
 
    def encode(self, x):
        x = x.to(device)
        h = self.encoder(x)
        h = h.view(-1, self.ndf*8*2*2) #.view returns a new tensor with a different shape, 4096
        if self.batchnorm:
            return self.bn_mean(self.fc1(h)), self.fc2(h)
        else:
            return self.fc1(h), self.fc2(h) #encoded z_mean, z_log_var
 
    def reparametrize(self, mu, logvar):
        print("start")
        std = logvar.mul(0.5).exp_()
        print("repara", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("hello", "using_GPU")
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        # eps = Variable(eps)
        eps = eps.cuda()
        print("eps", eps.is_cuda)
        return eps.mul(std).add_(mu)
 
    def decode(self, z):
        z=z.to(device)
        h = self.d1(z)
        h = h.view(-1, self.ngf*8, 2, 2)
        return self.decoder(h)
 
    def get_latent_var(self, x):
        x = x.to(device)
        print("x_lat", x.is_cuda)
        mu, logvar = self.encode(x.view(-1, self.nc, self.imsize, self.imsize))
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        z=z.to(device)
        print("z", z.is_cuda)
        res = self.decode(z)
        print("res", res.is_cuda)
        return res
 
    def forward(self, x):
        x=x.to(device)
        print("X", x.is_cuda)
        mu, logvar = self.encode(x.view(-1, self.nc, self.imsize, self.imsize))
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        print("res", res.is_cuda)
        return res, z, mu, logvar

class FC_VAE(nn.Module):
    """Fully connected variational Autoencoder"""
    def __init__(self, n_input, nz, n_hidden=1024): 
        super(FC_VAE, self).__init__()
        self.nz = nz #128
        self.n_input = n_input #60724
        self.n_hidden = n_hidden #1024

        self.encoder = nn.Sequential(nn.Linear(n_input, n_hidden), #60724*1024
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(n_hidden), #1024
                                nn.Linear(n_hidden, n_hidden), #1024*1024
                                nn.BatchNorm1d(n_hidden), #1024
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden), #1024*1024
                                nn.BatchNorm1d(n_hidden), #1024
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden), #1024*1024
                                nn.BatchNorm1d(n_hidden), #1024
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden), #1024*1024 = 1048576
                                )

        self.fc1 = nn.Linear(n_hidden, nz) #1024*128 = 131072
        self.fc2 = nn.Linear(n_hidden, nz)

        self.decoder = nn.Sequential(nn.Linear(nz, n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_input),
                                    )
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        res = self.decode(z)
        print("res", res.is_cuda)
        return res, z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        return self.fc1(h), self.fc2(h)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        # else:
        #     eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        return self.decoder(z)

    def get_latent_var(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return z
 
    def generate(self, z):
        res = self.decode(z)
        return res

class FC_Autoencoder(nn.Module):
    """Autoencoder"""
    def __init__(self, n_input, nz, n_hidden=512):
        super(FC_Autoencoder, self).__init__()
        self.nz = nz
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.encoder = nn.Sequential(nn.Linear(n_input, n_hidden),
                                nn.ReLU(inplace=True),
                                nn.BatchNorm1d(n_hidden),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, n_hidden),
                                nn.BatchNorm1d(n_hidden),
                                nn.ReLU(inplace=True),
                                nn.Linear(n_hidden, nz),
                                )

        self.decoder = nn.Sequential(nn.Linear(nz, n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_hidden),
                                     nn.BatchNorm1d(n_hidden),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(n_hidden, n_input),
                                    )

    def forward(self, x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return encoding, decoding

class FC_Classifier(nn.Module):
    """Latent space discriminator"""
    def __init__(self, nz, n_hidden=1024, n_out=2):
        super(FC_Classifier, self).__init__()
        self.nz = nz
        self.n_hidden = n_hidden
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
#            nn.Linear(n_hidden, n_hidden),
#            nn.ReLU(inplace=True),
 #           nn.Linear(n_hidden, n_hidden),
 #           nn.ReLU(inplace=True),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden,n_out)
        )

    def forward(self, x):
        return self.net(x)

class Simple_Classifier(nn.Module):
    """Latent space discriminator"""
    def __init__(self, nz, n_out=2):
        super(Simple_Classifier, self).__init__()
        self.nz = nz
        self.n_out = n_out

        self.net = nn.Sequential(
            nn.Linear(nz, n_out),
        )

    def forward(self, x):
        return self.net(x)

