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


cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

generator = G(opt)
generator.cuda()
generator.load_state_dict(torch.load('./models/generator.pkl'))

N=100

for i in range(opt.n_perf_classes):
  for j in range(opt.n_params_classes):
    print('Generate the architectures of param label %d, perf param %d...\n' % (j, i))
    z = FloatTensor(np.random.normal(0, 1, (N, 100)))
    gen_perf_labels = LongTensor(np.ones(N)*i)
    gen_param_labels = LongTensor(np.ones(N)*j)
    gen_conv_edges, gen_conv_nodes, gen_conv_ns, gen_reduc_edges, gen_reduc_nodes, gen_reduc_ns = generator(z, gen_perf_labels, gen_param_labels)
    gen_conv_archs = graph2arch(gen_conv_edges.data, gen_conv_nodes.data, gen_conv_ns.data)
    gen_reduc_archs = graph2arch(gen_reduc_edges.data, gen_reduc_nodes.data, gen_reduc_ns.data)
    perf_label_list = gen_perf_labels.data.cpu().numpy().tolist()
    param_label_list = gen_param_labels.data.cpu().numpy().tolist()
    spec_list = []
    assert len(gen_conv_archs) == len(gen_reduc_archs)
    for gen_conv_arch, gen_reduc_arch in zip(gen_conv_archs, gen_reduc_archs):
      conv_model_spec = ModelSpec(matrix=gen_conv_arch[0], ops=gen_conv_arch[1])
      reduc_model_spec = ModelSpec(matrix=gen_reduc_arch[0], ops=gen_reduc_arch[1])
      spec_list.append((conv_model_spec, reduc_model_spec))
    save_arch(spec_list, gen_perf_labels, gen_param_labels, None, 'archs/%d_%d.txt' % (j, i))
    torch.save(spec_list, 'archs_%d_%d' % (j, i))
