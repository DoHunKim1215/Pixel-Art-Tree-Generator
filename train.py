import os

import torch
from torch.backends import cudnn
from tqdm import tqdm

from models.arch import Generator, Discriminator
from models.loss import compute_gradient_penalty
from preprocess.data_loader import train_dataloader
from utils.adder import Adder
from utils.argument_manager import ArgumentManager
from utils.loss_example_plot import LossExamplePlot

if __name__ == '__main__':
    args = ArgumentManager.get_train_args()

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generator
    G = Generator(z_dim=args.noise_len).to(device)
    optimizer_G = torch.optim.Adam(G.parameters(), lr=args.gen_lr, betas=(args.beta1, args.beta2))

    G.train()
    for p in G.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)

    # Discriminator
    D = Discriminator().to(device)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=args.disc_lr, betas=(args.beta1, args.beta2))

    D.train()
    for p in D.parameters():
        torch.nn.init.normal_(p, mean=0.0, std=0.02)

    # Dataloader
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)

    # GPU Inference Timer
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # Loss Adder
    loss_G_adder = Adder()
    loss_D_adder = Adder()
    dist_W_adder = Adder()

    # Loss List
    loss_G_list = list()
    loss_D_list = list()
    dist_W_list = list()
    iter_list = list()

    one = torch.tensor(1, device=device, dtype=torch.float)
    minus_one = torch.tensor(-1, device=device, dtype=torch.float)

    epoch = 1
    iteration = 1

    # Resuming
    if args.resume is not None and os.path.isfile(args.resume):
        state = torch.load(args.resume)
        epoch = state['epoch']
        optimizer_G.load_state_dict(state['optimizer_G'])
        G.load_state_dict(state['model_G'])
        optimizer_D.load_state_dict(state['optimizer_D'])
        D.load_state_dict(state['model_D'])
        print('Resume from %d' % epoch)
        epoch += 1

    # Ready to Plot Loss Graph and Inference Example
    plot = LossExamplePlot()

    for epoch_idx in range(epoch, args.num_epoch + 1):

        starter.record()

        # Train Discriminator
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'Epoch {epoch_idx}', ncols=100,
                  leave=True) as pbar:
            for iter_idx, batch in pbar:
                img, label, name = batch
                img = img.to(device)
                label = label.to(device)

                batch_size = img.shape[0]

                iter_D_loss = 0
                iter_W_dist = 0

                # Reset requires_grad
                for p in D.parameters():
                    p.requires_grad = True

                # Train Discriminator
                for disc_iter in range(args.disc_freq):
                    D.zero_grad()

                    prob_real_is_real = D(img, label)
                    loss_D_real = prob_real_is_real.mean()
                    loss_D_real.backward(minus_one)

                    noise = torch.randn(batch_size, args.noise_len).uniform_(0, 1).to(device)

                    # Don't Train Generator
                    with torch.no_grad():
                        fake_img = G(noise, label)

                    fake_img_detached = fake_img.detach()
                    prob_fake_is_real = D(fake_img_detached, label)
                    loss_D_fake = prob_fake_is_real.mean()
                    loss_D_fake.backward(one)

                    gradient_penalty = compute_gradient_penalty(D, img.detach(), fake_img_detached, label.detach())
                    gradient_penalty *= args.lambda_gp
                    gradient_penalty.backward()
                    optimizer_D.step()

                    D_cost = loss_D_fake - loss_D_real + gradient_penalty
                    iter_D_loss += D_cost.detach().cpu().numpy()
                    iter_W_dist += (loss_D_real - loss_D_fake).detach().cpu().numpy()

                # Don't Train Discriminator
                for p in D.parameters():
                    p.requires_grad = False

                # Train Generator
                G.zero_grad()

                noise = torch.randn(batch_size, args.noise_len).uniform_(0, 1).to(device)
                fake_img = G(noise, label)
                prob_fake_is_real = D(fake_img, label)
                loss_G = prob_fake_is_real.mean()
                loss_G.backward(minus_one)
                optimizer_G.step()

                G_cost = -loss_G

                loss_G_adder(G_cost.detach().cpu().numpy())
                loss_D_adder(iter_D_loss / args.disc_freq)
                dist_W_adder(iter_W_dist / args.disc_freq)

                iter_list.append(iteration)
                loss_G_list.append(G_cost.detach().cpu().numpy())
                loss_D_list.append(iter_D_loss / args.disc_freq)
                dist_W_list.append(iter_W_dist / args.disc_freq)

                iteration += 1

        ender.record()
        torch.cuda.synchronize()
        elapsed = starter.elapsed_time(ender)

        # After one epoch, save model weights
        overwrite_name = os.path.join(args.model_out_dir, 'model.pkl')
        torch.save({'model_G': G.state_dict(),
                    'optimizer_G': optimizer_G.state_dict(),
                    'model_D': D.state_dict(),
                    'optimizer_D': optimizer_D.state_dict(),
                    'epoch': epoch_idx}, overwrite_name)

        # After save_freq epoch, save model weights in other file
        if epoch_idx % args.save_freq == 0:
            save_name = os.path.join(args.model_out_dir, 'model_%d.pkl' % epoch_idx)
            torch.save({'model_G': G.state_dict(),
                        'optimizer_G': optimizer_G.state_dict(),
                        'model_D': D.state_dict(),
                        'optimizer_D': optimizer_D.state_dict(),
                        'epoch': epoch_idx}, save_name)

        print(
            "EPOCH: %02d\nElapsed time(s): %4.2f Epoch Loss(G): %7.6f Epoch Loss(D): %7.6f Wasserstein Distance: %7.6f" % (
                epoch_idx, elapsed / 1000., loss_G_adder.average(), loss_D_adder.average(), dist_W_adder.average()))

        # Plot Generator's Samples
        plot.plot_leaf_example(G, args.noise_len, device)
        plot.plot_fruit_example(G, args.noise_len, device)
        plot.plot_trunk_example(G, args.noise_len, device)

        plot.plot_losses(iter_list, loss_G_list, loss_D_list, dist_W_list)

        plot.update()
        plot.save(os.path.join(args.fig_out_dir, 'fig_%d.png' % epoch_idx))

        loss_G_adder.reset()
        loss_D_adder.reset()
        dist_W_adder.reset()

    plot.disable_interactive()

    save_name = os.path.join(args.model_out_dir, 'Final.pkl')
    torch.save({'model_G': G.state_dict(), 'model_D': D.state_dict()}, save_name)
