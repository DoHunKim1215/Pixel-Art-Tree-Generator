import torch
from torch import nn, Tensor


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, batch_norm: bool = True):
        super(LinearBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.BatchNorm1d(out_features) if batch_norm else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (batch_size, in_features)
        :return: (batch_size, out_features)
        """
        return self.main(x)


class DownConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownConvBlock, self).__init__()

        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (B, in_channels, H, W)
        :return: (B, out_channels, H / 2, W / 2)
        """
        return self.layer0(x)


class UpConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpConvBlock, self).__init__()

        self.layer0 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: (B, in_channels, H, W)
        :return: (B, out_channels, H * 2, W * 2)
        """
        return self.layer0(x)


class Generator(nn.Module):
    def __init__(self, z_dim=64, base_channels=128):
        super(Generator, self).__init__()

        self.base_channels = base_channels

        self.embedding = nn.ModuleList([
            nn.Embedding(num_embeddings=8, embedding_dim=8 // 2),  # Leaf Type
            nn.Embedding(num_embeddings=6, embedding_dim=6 // 2),  # Trunk Type
            nn.Embedding(num_embeddings=7, embedding_dim=7 // 2)  # Fruit Type
        ])

        self.projection = nn.ModuleList([
            LinearBlock(in_features=16, out_features=z_dim),
            LinearBlock(in_features=2 * z_dim, out_features=16 * base_channels),
        ])

        self.hidden = nn.Sequential(
            UpConvBlock(16 * base_channels, 8 * base_channels),
            UpConvBlock(8 * base_channels, 4 * base_channels),
            UpConvBlock(4 * base_channels, 2 * base_channels),
            UpConvBlock(2 * base_channels, base_channels),
        )

        self.output = nn.Conv2d(in_channels=base_channels, out_channels=4, kernel_size=1, padding=0)
        self.tanh = nn.Tanh()

    def forward(self, z, x):
        """
        :param z: (batch_size, z_dim)
        :param x: (batch_size, 9)
        :return: (batch_size, 4, 16, 16)
        """
        # Input
        y = torch.cat([
            self.embedding[0](x[:, 0]),  # Leaf Type
            x[:, 1:4] / 255.0,  # Leaf Color
            self.embedding[1](x[:, 4]),  # Trunk Type
            self.embedding[2](x[:, 5]),  # Fruit Type
            x[:, 6:] / 255.0  # Fruit Color
        ], dim=1)
        y = self.projection[1](torch.cat([self.projection[0](y), z], dim=1))
        y = y.view(-1, 16 * self.base_channels, 1, 1)

        # Hidden
        y = self.hidden(y)

        # Output
        y = self.output(y)

        # Activate
        return self.tanh(y)


class Discriminator(nn.Module):
    def __init__(self, base_channels=128):
        super(Discriminator, self).__init__()

        self.base_channels = base_channels

        self.embedding = nn.ModuleList([
            nn.Embedding(num_embeddings=8, embedding_dim=8 // 2),
            nn.Embedding(num_embeddings=6, embedding_dim=6 // 2),
            nn.Embedding(num_embeddings=7, embedding_dim=7 // 2)
        ])

        self.projection = LinearBlock(in_features=16, out_features=16 * 16 * 4, batch_norm=False)

        self.input = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=base_channels, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.hidden = nn.Sequential(
            DownConvBlock(base_channels, 2 * base_channels),
            DownConvBlock(2 * base_channels, 4 * base_channels),
            DownConvBlock(4 * base_channels, 8 * base_channels),
            DownConvBlock(8 * base_channels, 16 * base_channels),
        )

        self.output = nn.Linear(in_features=16 * base_channels, out_features=1)

        # No Activation Function

    def forward(self, x, label):
        """
        :param x: (batch_size, 4, 16, 16)
        :param label: (batch_size, 9)
        :return: (batch_size, 1)
        """
        # Input
        y = torch.cat([
            self.embedding[0](label[:, 0]),  # Leaf Type
            label[:, 1:4] / 255.0,  # Leaf Color
            self.embedding[1](label[:, 4]),  # Trunk Type
            self.embedding[2](label[:, 5]),  # Fruit Type
            label[:, 6:] / 255.0  # Fruit Color
        ], dim=1)
        y = self.projection(y).view(-1, 4, 16, 16)

        # Hidden
        y = self.input(torch.cat([x, y], dim=1))
        z = self.hidden(y)

        # Output
        z = z.view(-1, 16 * self.base_channels)
        z = self.output(z)

        # No Activation
        return z
