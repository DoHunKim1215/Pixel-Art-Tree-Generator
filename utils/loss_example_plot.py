import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from torch import nn

from utils.dataset_analyzer import denormalize


class LossExamplePlot():
    leaf_types = {0: 'acacia', 1: 'bush', 2: 'shrub', 3: 'pine', 4: 'oak', 5: 'palm', 6: 'poplar', 7: 'willow'}
    trunk_types = {0: 'oak', 1: 'slime', 2: 'swamp', 3: 'cherry', 4: 'old', 5: 'jungle'}
    fruit_types = {0: 'circle', 1: 'hanging', 2: 'berry', 3: 'long', 4: 'star', 5: 'pop', 6: 'fruitless'}

    def __init__(self):
        plt.ion()
        self.fig = plt.figure(figsize=(16, 16))

        # Loss Graph
        self.ax = self.fig.add_subplot(221)
        self.line1, = self.ax.plot(list(), list(), color='orange', label='Generator Loss')
        self.line2, = self.ax.plot(list(), list(), color='blue', label='Discriminator Loss')
        self.line3, = self.ax.plot(list(), list(), color='red', label='Wasserstein Distance')

        self.ax.set_xlabel('Iterations')
        self.ax.set_title('Loss of Generator and Discriminator')
        self.ax.legend(handles=[self.line1, self.line2, self.line3], loc='upper left')

        # Example Plot
        self.leaf_color_type_grid = ImageGrid(self.fig, 222, nrows_ncols=(8, 8), axes_pad=0.1, share_all=True)
        self.leaf_color_type_grid.axes_llc.set(xticks=[], yticks=[])
        for i in range(8):
            self.leaf_color_type_grid[i].set_xlabel(LossExamplePlot.leaf_types[i], rotation=0)

        self.fruit_color_type_grid = ImageGrid(self.fig, 223, nrows_ncols=(8, 7), axes_pad=0.1, share_all=True)
        self.fruit_color_type_grid.axes_llc.set(xticks=[], yticks=[])
        for i in range(7):
            self.fruit_color_type_grid[i].set_xlabel(LossExamplePlot.fruit_types[i], rotation=0)

        self.trunk_type_grid = ImageGrid(self.fig, 224, nrows_ncols=(8, 6), axes_pad=0.1, share_all=True)
        self.trunk_type_grid.axes_llc.set(xticks=[], yticks=[])
        for i in range(6):
            self.trunk_type_grid[i].set_xlabel(LossExamplePlot.trunk_types[i], rotation=0)

    def plot_losses(self, iter_list, loss_G_list, loss_D_list, dist_W_list):
        self.line1.set_data(iter_list, loss_G_list)
        self.line2.set_data(iter_list, loss_D_list)
        self.line3.set_data(iter_list, dist_W_list)

        self.ax.relim()
        self.ax.autoscale_view()

    def plot_leaf_example(self, G: nn.Module, noise_dim: int, device: torch.device):
        colors = [
            torch.tensor([[0xF9, 0x41, 0x44]], device=device),
            torch.tensor([[0xF3, 0x72, 0x2C]], device=device),
            torch.tensor([[0xF9, 0xC7, 0x4F]], device=device),
            torch.tensor([[0x90, 0xBE, 0x6D]], device=device),
            torch.tensor([[0x43, 0xAA, 0x8D]], device=device),
            torch.tensor([[0x4D, 0x90, 0x8E]], device=device),
            torch.tensor([[0x57, 0x75, 0x90]], device=device),
            torch.tensor([[0x27, 0x7D, 0xA1]], device=device),
        ]

        G.eval()

        with torch.no_grad():
            x = 0
            y = 0
            for ax_grid in self.leaf_color_type_grid:
                noise = torch.randn(1, noise_dim).uniform_(0, 1).to(device)
                leaf_type = torch.full([1, 1], x).to(device)
                leaf_color = colors[y]
                trunk_type = torch.full([1, 1], 3).to(device)
                fruit_type = torch.full([1, 1], 6).to(device)
                fruit_color = torch.tensor([[255, 255, 255]]).to(device)

                feature = torch.cat([leaf_type, leaf_color, trunk_type, fruit_type, fruit_color], dim=1)

                fake = G(noise, feature).squeeze(0).cpu()
                fake_rgb = denormalize(fake[0:3, :, :], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                fake_alpha = torch.heaviside(fake[3:4, :, :], values=torch.tensor([1.0])).to(
                    torch.float).numpy().transpose(1, 2, 0)

                ax_grid.clear()
                ax_grid.imshow(np.concatenate([fake_rgb, fake_alpha], axis=-1))

                if x == 7:
                    x = 0
                    y += 1
                else:
                    x += 1

        G.train()

    def plot_fruit_example(self, G: nn.Module, noise_dim: int, device: torch.device):
        colors = [
            torch.tensor([[0xF9, 0x41, 0x44]], device=device),
            torch.tensor([[0xF3, 0x72, 0x2C]], device=device),
            torch.tensor([[0xF9, 0xC7, 0x4F]], device=device),
            torch.tensor([[0x90, 0xBE, 0x6D]], device=device),
            torch.tensor([[0x43, 0xAA, 0x8D]], device=device),
            torch.tensor([[0x4D, 0x90, 0x8E]], device=device),
            torch.tensor([[0x57, 0x75, 0x90]], device=device),
            torch.tensor([[0x27, 0x7D, 0xA1]], device=device),
        ]

        G.eval()

        with torch.no_grad():
            x = 0
            y = 0
            for ax_grid in self.fruit_color_type_grid:
                noise = torch.randn(1, noise_dim).uniform_(0, 1).to(device)
                leaf_type = torch.full([1, 1], 4).to(device)
                leaf_color = torch.tensor([[0x90, 0xBE, 0x6D]], device=device)
                trunk_type = torch.full([1, 1], 3).to(device)
                fruit_type = torch.full([1, 1], x).to(device)
                fruit_color = colors[y]

                feature = torch.cat([leaf_type, leaf_color, trunk_type, fruit_type, fruit_color], dim=1)

                fake = G(noise, feature).squeeze(0).cpu()
                fake_rgb = denormalize(fake[0:3, :, :], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                fake_alpha = torch.heaviside(fake[3:4, :, :], values=torch.tensor([1.0])).to(
                    torch.float).numpy().transpose(1, 2, 0)

                ax_grid.clear()
                ax_grid.imshow(np.concatenate([fake_rgb, fake_alpha], axis=-1))

                if x == 6:
                    x = 0
                    y += 1
                else:
                    x += 1

        G.train()

    def plot_trunk_example(self, G: nn.Module, noise_dim: int, device: torch.device):
        G.eval()

        with torch.no_grad():
            x = 0
            y = 0
            for ax_grid in self.trunk_type_grid:
                noise = torch.randn(1, noise_dim).uniform_(0, 1).to(device)
                leaf_type = torch.randint(0, 7, torch.Size([1, 1])).to(device)
                leaf_color = torch.randint(0, 255, torch.Size([1, 3])).to(device)
                trunk_type = torch.full([1, 1], x).to(device)
                fruit_type = torch.randint(0, 6, torch.Size([1, 1])).to(device)
                fruit_color = torch.randint(0, 255, torch.Size([1, 3])).to(device)

                feature = torch.cat([leaf_type, leaf_color, trunk_type, fruit_type, fruit_color], dim=1)

                fake = G(noise, feature).squeeze(0).cpu()
                fake_rgb = denormalize(fake[0:3, :, :], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                fake_alpha = torch.heaviside(fake[3:4, :, :], values=torch.tensor([1.0])).to(
                    torch.float).numpy().transpose(1, 2, 0)

                ax_grid.clear()
                ax_grid.imshow(np.concatenate([fake_rgb, fake_alpha], axis=-1))

                if x == 5:
                    x = 0
                    y += 1
                else:
                    x += 1

        G.train()

    def update(self):
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def save(self, path):
        plt.savefig(path)

    def disable_interactive(self):
        plt.ioff()
        plt.show()
