import torch
from torch import nn, Tensor


def compute_gradient_penalty(discriminator: nn.Module, real_samples: Tensor, fake_samples: Tensor, label: Tensor):
    batch_size = real_samples.shape[0]
    alpha = torch.randn(batch_size, 1, 1, 1).cuda()
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).cuda().requires_grad_(True)

    d_interpolates = discriminator(interpolates, label)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).cuda(),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty
