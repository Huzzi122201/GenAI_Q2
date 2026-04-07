import torch
import torch.nn as nn

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x, skip):
        x = self.model(x)
        x = torch.cat((x, skip), dim=1)
        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()
        
        # Encoder
        self.down1 = UNetDownBlock(in_channels, features, normalize=False)
        self.down2 = UNetDownBlock(features, features*2)
        self.down3 = UNetDownBlock(features*2, features*4)
        self.down4 = UNetDownBlock(features*4, features*8, dropout=0.5)
        self.down5 = UNetDownBlock(features*8, features*8, dropout=0.5)
        self.down6 = UNetDownBlock(features*8, features*8, dropout=0.5)
        self.down7 = UNetDownBlock(features*8, features*8, dropout=0.5)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, 4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.up1 = UNetUpBlock(features*8, features*8, dropout=0.5)
        self.up2 = UNetUpBlock(features*16, features*8, dropout=0.5)
        self.up3 = UNetUpBlock(features*16, features*8, dropout=0.5)
        self.up4 = UNetUpBlock(features*16, features*8, dropout=0.5)
        self.up5 = UNetUpBlock(features*16, features*4)
        self.up6 = UNetUpBlock(features*8, features*2)
        self.up7 = UNetUpBlock(features*4, features)
        
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        
        # Bottleneck
        bottleneck = self.bottleneck(d7)
        
        # Decoder with skip connections
        u1 = self.up1(bottleneck, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels*2, features, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(features, features*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(features*2, features*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(features*4, features*8, 4, stride=1, padding=1),
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(features*8, 1, 4, stride=1, padding=1)
        )
    
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        return self.model(xy)