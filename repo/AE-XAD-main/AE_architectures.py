import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from torchvision.models import resnet50
import torch.nn.functional as F


class Shallow_Autoencoder(nn.Module):
    def __init__(self, dim, flat_dim, latent_dim):
        super(Shallow_Autoencoder, self).__init__()
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat_dim),
            nn.Sigmoid(),
        )


    def forward(self, x):
        x_f = x.flatten(start_dim=1)
        encoded = self.encoder(x_f)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, x.shape)
        return decoded


class Deep_Autoencoder(nn.Module):
    def __init__(self, dim, flat_dim, intermediate_dim, latent_dim):
        super(Deep_Autoencoder, self).__init__()
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            nn.ReLU(),
            nn.Linear(intermediate_dim, flat_dim),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x_f = x.flatten(start_dim=1)
        encoded = self.encoder(x_f)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, x.shape)
        return decoded


class Conv_Deep_Autoencoder(nn.Module):
    def __init__(self, dim):
        super(Conv_Deep_Autoencoder, self).__init__()
        self.dim = np.array(dim)
        print(dim)

        layers = []
        diffs = []

        layers.append(nn.Conv2d(dim[0], 16, (5, 5), stride=1, padding=2))
        mods = np.remainder(-self.dim[1:], 4)
        diff = np.array([mods[-2] // 2, mods[-2] - mods[-2] // 2, mods[-1] // 2, mods[-1] - mods[-1] // 2])
        if np.sum(diff) > 0:
            layers.append(nn.ZeroPad2d(diff))
        diffs.append(diff)
        layers.append(nn.MaxPool2d((4, 4), stride=4))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 32, (5, 5), stride=1, padding=2))
        mods = np.remainder(-np.floor_divide(self.dim[1:], 4), 4)
        diff = np.array([mods[-2] // 2, mods[-2] - mods[-2] // 2, mods[-1] // 2, mods[-1] - mods[-1] // 2])
        if np.sum(diff) > 0:
            layers.append(nn.ZeroPad2d(diff))
        diffs.append(diff)
        layers.append(nn.MaxPool2d((4, 4), stride=4))
        layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)

        layers = []

        if np.sum(diffs[0]) > 0:
            layers.append(nn.ZeroPad2d(-diffs[0]))
        layers.append(nn.Upsample(scale_factor=4))
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.ReLU())
        if np.sum(diffs[1]) > 0:
            layers.append(nn.ZeroPad2d(-diffs[1]))
        layers.append(nn.Upsample(scale_factor=4))
        layers.append(nn.Conv2d(32, 16, (5, 5), stride=1, padding=2))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, dim[0], (5, 5), padding=2))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class Conv_Deep_Autoencoder_v2(nn.Module):
    def __init__(self, dim):
        super(Conv_Deep_Autoencoder_v2, self).__init__()
        self.dim = np.array(dim)
        print(dim)

        layers = []

        layers.append(nn.Conv2d(dim[0], 16, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 32, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 128, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())



        self.encoder = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(128, 32, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(32, 32, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(32))
        layers.append(nn.ReLU())

        layers.append(nn.ConvTranspose2d(32, 16, (6, 6), stride=4, padding=1))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, 16, (5, 5), stride=1, padding=2))
        layers.append(nn.BatchNorm2d(16))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(16, dim[0], (3, 3), stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




class Conv_Autoencoder(nn.Module):
    def __init__(self, dim):
        super(Conv_Autoencoder, self).__init__()
        self.dim = dim
        print(dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(dim[0], 16, (3, 3), stride=1, padding=1), #In torch non c'è padding='same' come tensorflow per strides > 2
            nn.MaxPool2d((2,2), stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 8, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2), stride=2),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 8, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            #nn.Conv2d(8, 8, (3, 3), padding=0),
            #nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(8, 16, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, dim[0], (3, 3), padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class PCA_Autoencoder(nn.Module):
    def __init__(self, dim, flat_dim, latent_dim):
        super(PCA_Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.dim = dim

        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, latent_dim, bias=False),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat_dim, bias=False),
        )

    def forward(self, x):
        x_f = x.flatten(start_dim=1)
        encoded = self.encoder(x_f)
        decoded = self.decoder(encoded)
        decoded = torch.reshape(decoded, x.shape)
        return decoded

class ResNet_CNN_Attn(nn.Module):
    def __init__(self, dim):
        super(ResNet_CNN_Attn, self).__init__()
        self.dim = dim
        encoder_pre = resnet50(pretrained=True)
        for param in encoder_pre.parameters():
            param.requires_grad = False

        self.conv1 = encoder_pre.conv1
        self.bn1 = encoder_pre.bn1
        self.relu1 = encoder_pre.relu
        self.maxpool1 = encoder_pre.maxpool
        self.layer1 = encoder_pre.layer1
        self.layer2 = encoder_pre.layer2

        layers = []

        layers.append(nn.Conv2d(512, 256, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(256, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(64))
        #layers.append(nn.ReLU())

        #self.att = CBAM(64)

        self.encoder = nn.Sequential(*layers)

        self.up1 = nn.Upsample(scale_factor=2)

        self.up2 = nn.Upsample(scale_factor=2)

        self.up3 = nn.Upsample(scale_factor=2)
        self.tan3 = nn.Tanh()

        layers = []

        layers.append(nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec1 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(32, 16, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(16, 16, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec2 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(16, 8, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec3 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        # layers.append(nn.BatchNorm2d(16))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, dim[0], (3, 3), stride=1, padding=1))
        #layers.append(nn.Sigmoid())
        layers.append(nn.ReLU())

        self.decoder_final = nn.Sequential(*layers)

    def stop_finetuning(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.maxpool1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False


    def forward(self, x):
        fe = self.conv1(x)
        fe = self.bn1(fe)
        fe = self.relu1(fe)
        fe = self.maxpool1(fe)
        fe1 = self.layer1(fe)
        fe2 = self.layer2(fe1)

        encoded = self.encoder(fe2)

        upsampled1 = self.up1(encoded)
        decoded1 = self.dec1(encoded)

        upsampled2 = self.up2(upsampled1)
        decoded2 = self.dec2(decoded1)

        upsampled3 = self.up3(upsampled2)
        decoded3 = self.dec3(decoded2)
        decoded3 = decoded3 + decoded3 * torch.sum(self.tan3(upsampled3)**2, axis=1).unsqueeze(1)

        decoded = self.decoder_final(decoded3)  # unsqueeze

        return decoded


class ResNet_CNN_Attn_ablation(nn.Module):
    def __init__(self, dim):
        super(ResNet_CNN_Attn_ablation, self).__init__()
        self.dim = dim
        encoder_pre = resnet50(pretrained=True)
        for param in encoder_pre.parameters():
            param.requires_grad = False

        self.conv1 = encoder_pre.conv1
        self.bn1 = encoder_pre.bn1
        self.relu1 = encoder_pre.relu
        self.maxpool1 = encoder_pre.maxpool
        self.layer1 = encoder_pre.layer1
        self.layer2 = encoder_pre.layer2

        layers = []

        layers.append(nn.Conv2d(512, 256, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(256, 128, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(128))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(128, 64, (3, 3), stride=1, padding=1))
        layers.append(nn.BatchNorm2d(64))

        self.encoder = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(64, 32, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(32, 32, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec1 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(32, 16, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(16, 16, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec2 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.ConvTranspose2d(16, 8, (4, 4), stride=2, padding=1))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        layers.append(nn.SELU())

        self.dec3 = nn.Sequential(*layers)

        layers = []

        layers.append(nn.Conv2d(8, 8, (3, 3), stride=1, padding=1))
        # layers.append(nn.BatchNorm2d(16))
        layers.append(nn.SELU())
        layers.append(nn.Conv2d(8, dim[0], (3, 3), stride=1, padding=1))
        #layers.append(nn.Sigmoid())
        layers.append(nn.ReLU())

        self.decoder_final = nn.Sequential(*layers)

    def stop_finetuning(self):
        for param in self.conv1.parameters():
            param.requires_grad = False
        for param in self.bn1.parameters():
            param.requires_grad = False
        for param in self.maxpool1.parameters():
            param.requires_grad = False
        for param in self.layer1.parameters():
            param.requires_grad = False
        for param in self.layer2.parameters():
            param.requires_grad = False


    def forward(self, x):
        fe = self.conv1(x)
        fe = self.bn1(fe)
        fe = self.relu1(fe)
        fe = self.maxpool1(fe)
        fe1 = self.layer1(fe)
        fe2 = self.layer2(fe1)

        encoded = self.encoder(fe2)

        decoded1 = self.dec1(encoded)
        decoded2 = self.dec2(decoded1)
        decoded3 = self.dec3(decoded2)
        decoded3 = decoded3

        decoded = self.decoder_final(decoded3)  # unsqueeze

        return decoded








