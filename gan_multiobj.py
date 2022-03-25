import argparse
import os
import numpy as np
import math
import random

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.backends.cudnn as cudnn

from tensorboardX import SummaryWriter

from generator import G_MultiObj as G
from discriminator import D_MultiObj as D
from utils import NASBenchwithLabel, graph2arch, save_arch, sample_random, ModelSpec

os.makedirs("archs", exist_ok=True)
os.makedirs("models", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
parser.add_argument("--vocab_size", type=int, default=8, help="the number of op defined on nodes")
parser.add_argument("--g_emb_size", type=int, default=250, help="the emb_size of generator")
parser.add_argument("--d_emb_size", type=int, default=250, help="the emb_size of discriminator")
parser.add_argument("--g_num_layers", type=int, default=3, help="the number of gnn layers in generator")
parser.add_argument("--d_gnn_num_layers", type=int, default=3, help="the number of gnn layers in discriminator")
parser.add_argument("--d_mlp_num_layers", type=int, default=4, help="the number of mlp layers in discriminator")
parser.add_argument("--g_dropout", type=float, default=0.5, help="the dropout of gnn layers in generator")
parser.add_argument("--d_gnn_dropout", type=float, default=0.5, help="the dropout of gnn layers in discriminator")
parser.add_argument("--d_mlp_dropout", type=float, default=0.5, help="the dropout of mlp layers in discriminator")
parser.add_argument("--g_hidden_size", type=int, default=56, help="the size of the hidden layer in generator")
parser.add_argument("--d_hidden_size", type=int, default=56, help="the size of the hidden layer in discriminator")
parser.add_argument("--g_aggr", type=str, default='gsum', help="how to aggr the nodes in gnn")
parser.add_argument("--max_step", type=int, default=10, help="the max step in the cell of the network")
parser.add_argument("--nasbench_data", type=str, default='./data/nasdata', help="the max step in the cell of the network")
parser.add_argument("--n_critic", type=int, default=1, help="number of training steps for discriminator per iter")
parser.add_argument("--n_perf_classes", type=int, default=3, help="number of perf classes")
parser.add_argument("--cls_perf_embedding_size", type=int, default=10, help="size of perf classes embedding")
parser.add_argument("--n_params_classes", type=int, default=3, help="number of params classes")
parser.add_argument("--cls_params_embedding_size", type=int, default=10, help="size of params classes embedding")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
parser.add_argument("--seed", type=int, default=1997, help="the random seed")
parser.add_argument("--gpu", type=int, default=0, help="which gpu")
opt = parser.parse_args()
print(opt)


# tensorboardX writer
writer = SummaryWriter()

# Configure cuda and seed
cuda = True if torch.cuda.is_available() else False

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.set_device(opt.gpu)
    cudnn.benchmark = True
    cudnn.enabled=True
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

# Loss function
adversarial_loss = torch.nn.MSELoss()

# Initialize generator and discriminator
generator = G(opt)
print('The generator model:', generator)
total_params_g = sum(x.data.nelement() for x in generator.parameters())
print('Generator total parameters: {}'.format(total_params_g))

discriminator = D(opt)
print('The discriminator model:', discriminator)
total_params_d = sum(x.data.nelement() for x in discriminator.parameters())
print('Discriminator total parameters: {}'.format(total_params_d))

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

data = torch.load(opt.nasbench_data)
print('data size is {}'.format(len(data)))

ratios = [1.0]

batches_done=0
for ratio in ratios:
    # Configure data loader
    sampled_data = sample_random(data, ratio)
    sampled_data = sampled_data[:int(len(sampled_data)//opt.batch_size*opt.batch_size)]
    nasbench_dataset = NASBenchwithLabel(sampled_data, opt.max_step, opt.n_perf_classes, opt.n_params_classes)
    nasbench_dataloader = DataLoader(nasbench_dataset, batch_size=opt.batch_size, shuffle=True)

    # ----------
    #  Training
    # ----------

    for epoch in range(opt.n_epochs):
        if epoch%10==0:
            torch.save(generator.state_dict(), './models/generator_{}.pkl'.format(epoch))
            torch.save(discriminator.state_dict(), './models/discriminator_{}.pkl'.format(epoch))
        for i, (conv_edges, conv_nodes, conv_ns, reduc_edges, reduc_nodes, reduc_ns, perf_labels, param_labels) in enumerate(nasbench_dataloader):

            # Adversarial ground truths
            valid = FloatTensor(conv_edges.size(0), 1).fill_(0.0)
            fake = FloatTensor(conv_edges.size(0), 1).fill_(1.0)

            # Configure input
            real_conv_edges = conv_edges.type(FloatTensor)
            real_conv_nodes = conv_nodes.type(FloatTensor)
            real_conv_ns    = conv_ns.type(LongTensor)
            real_reduc_edges = reduc_edges.type(FloatTensor)
            real_reduc_nodes = reduc_nodes.type(FloatTensor)
            real_reduc_ns    = reduc_ns.type(LongTensor)
            real_perf_labels = perf_labels.type(LongTensor).squeeze()
            real_param_labels = param_labels.type(LongTensor).squeeze()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = FloatTensor(np.random.normal(0, 1, (conv_ns.shape[0], opt.latent_dim)))
            gen_perf_labels = LongTensor(np.random.randint(0, opt.n_perf_classes, conv_ns.shape[0]))
            gen_param_labels = LongTensor(np.random.randint(0, opt.n_params_classes, conv_ns.shape[0]))

            # Generate a batch of images
            gen_conv_edges, gen_conv_nodes, gen_conv_ns, gen_reduc_edges, gen_reduc_nodes, gen_reduc_ns = generator(z, gen_perf_labels, gen_param_labels)

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(real_conv_edges, real_conv_nodes, real_conv_ns, real_reduc_edges, real_reduc_nodes, real_reduc_ns, real_perf_labels, real_param_labels), valid)
            fake_loss = adversarial_loss(discriminator(gen_conv_edges.detach(), gen_conv_nodes.detach(), gen_conv_ns.detach(), gen_reduc_edges.detach(), gen_reduc_nodes.detach(), gen_reduc_ns.detach(), gen_perf_labels.detach(), gen_param_labels.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            #nn.utils.clip_grad_norm_(discriminator.parameters(), opt.clip_value)
            optimizer_D.step()

            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = FloatTensor(np.random.normal(0, 1, (conv_ns.shape[0], opt.latent_dim)))
                gen_perf_labels = LongTensor(np.random.randint(0, opt.n_perf_classes, conv_ns.shape[0]))
                gen_param_labels = LongTensor(np.random.randint(0, opt.n_params_classes, conv_ns.shape[0]))

                # Generate a batch of images
                gen_conv_edges, gen_conv_nodes, gen_conv_ns, gen_reduc_edges, gen_reduc_nodes, gen_reduc_ns = generator(z, gen_perf_labels, gen_param_labels)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_conv_edges, gen_conv_nodes, gen_conv_ns, gen_reduc_edges, gen_reduc_nodes, gen_reduc_ns, gen_perf_labels, gen_param_labels), valid)

                g_loss.backward()
                #nn.utils.clip_grad_norm_(generator.parameters(), opt.clip_value)
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch+1, opt.n_epochs, i, len(nasbench_dataloader), d_loss.item(), g_loss.item())
                )
                writer.add_scalar('loss/d_loss', d_loss, batches_done)
                writer.add_scalar('loss/g_loss', g_loss, batches_done)

            batches_done += 1

        if (epoch+1) % 2 == 0:
            num_logging = None
            gen_conv_archs = graph2arch(gen_conv_edges.data[:num_logging], gen_conv_nodes.data[:num_logging], gen_conv_ns.data[:num_logging])
            gen_reduc_archs = graph2arch(gen_reduc_edges.data[:num_logging], gen_reduc_nodes.data[:num_logging], gen_reduc_ns.data[:num_logging])
            perf_label_list = gen_perf_labels.data.cpu().numpy().tolist()
            param_label_list = gen_param_labels.data.cpu().numpy().tolist()
            spec_list = []
            assert len(gen_conv_archs) == len(gen_reduc_archs)
            for gen_conv_arch, gen_reduc_arch in zip(gen_conv_archs, gen_reduc_archs):
                conv_model_spec = ModelSpec(matrix=gen_conv_arch[0], ops=gen_conv_arch[1])
                reduc_model_spec = ModelSpec(matrix=gen_reduc_arch[0], ops=gen_reduc_arch[1])
                spec_list.append((conv_model_spec, reduc_model_spec))
            save_arch(spec_list, perf_label_list, param_label_list, None, 'archs/%d.txt' % batches_done)
            torch.save(spec_list, 'archs/archs_%d' % batches_done)
            '''for name, param in discriminator.named_parameters():
                writer.add_histogram('D %s' % name, param.clone().cpu().data.numpy(), batches_done)
            for name, param in generator.named_parameters():
                writer.add_histogram('G %s' % name, param.clone().cpu().data.numpy(), batches_done)
            for name, param in discriminator.named_parameters():
                writer.add_scalar('grad/D %s grad' % name, np.linalg.norm(param.grad.clone().cpu().data.numpy(), 2), batches_done)
            for name, param in generator.named_parameters():
                writer.add_scalar('grad/G %s grad' % name, np.linalg.norm(param.grad.clone().cpu().data.numpy(), 2), batches_done)'''

writer.close()

torch.save(generator.state_dict(), './models/generator.pkl')
torch.save(discriminator.state_dict(), './models/discriminator.pkl')
