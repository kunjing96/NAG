import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout, act=F.leaky_relu, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    '''def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, mode='fan_out')
        if self.bias is not None:
            nn.init.kaiming_uniform_(self.bias, mode='fan_out')'''

    def forward(self, adj, input):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            output = output + self.bias
        if self.act is not None:
            output = self.act(output, negative_slope=0.2)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class NodeEmb(nn.Module):
    def __init__(self, vocab_size, emb_size, hid_size, num_layers, dropout):
        super(NodeEmb,self).__init__()
        self.embedding = nn.Linear(vocab_size, emb_size, bias=False)
        self.gcns = nn.ModuleList([GraphConvolution(emb_size if i==0 else hid_size, hid_size, dropout=dropout, act= None if i==num_layers-1 else F.leaky_relu) for i in range(num_layers)])

    def norm(self, adj):
        adj = adj + adj.transpose(1, 2) + torch.zeros_like(adj).scatter_(-1, torch.LongTensor(range(adj.size(1))).unsqueeze(-1).unsqueeze(0).repeat(adj.size(0), 1, 1).cuda(), 1)
        degrees = torch.pow(adj.sum(-1).float(), -0.5)
        D = torch.cat([torch.diag(x).unsqueeze(0) for x in degrees], dim=0)
        return torch.matmul(torch.matmul(adj, D).transpose(1, 2), D)

    def forward(self, adj, input):
        h = self.embedding(input)
        norm_adj = self.norm(adj)
        for i in range(len(self.gcns)):
            h = self.gcns[i](norm_adj, h)
        return F.normalize(h, 2, dim=-1)


class GraphEmb(nn.Module):
    def __init__(self, node_emb_size, graph_emb_size, aggr='gsum'):
        super(GraphEmb, self).__init__()
        self.aggr = aggr
        self.f_m = nn.Linear(node_emb_size, graph_emb_size) 
        if aggr == 'gsum': 
            self.g_m = nn.Linear(node_emb_size, 1) 
            self.sigm = nn.Sigmoid()

    def forward(self, h, ns):
        batch_size = h.size(0)
        if self.aggr == 'mean':
            h = self.f_m(h)
            return F.normalize(torch.cat([torch.mean(h[i, :ns[i], :], dim=0).unsqueeze(0) for i in range(batch_size)], dim=0).contiguous(), 2, dim=-1)
        elif self.aggr == 'gsum':
            h_vG = self.f_m(h)
            g_vG = self.sigm(self.g_m(h))
            h_G = torch.mul(h_vG, g_vG)
            return F.normalize(torch.cat([torch.sum(h_G[i, :ns[i], :], dim=0).unsqueeze(0) for i in range(batch_size)], dim=0).contiguous(), 2, dim=-1)


class GetAcc(nn.Module): 
    def __init__(self, graph_emb_size, num_layers, dropout):
        super(GetAcc, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.lin_layers = nn.ModuleList([nn.Linear(graph_emb_size//(2**num), graph_emb_size//(2**(num+1))) for num in range(num_layers-1)]) 
        self.lin_layers.append(nn.Linear(graph_emb_size//(2**(num_layers-1)), 1))

    def forward(self, h):
        for layer in self.lin_layers[:-1]:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = F.leaky_relu(layer(h), negative_slope=0.2) 
        h = self.lin_layers[-1](h)
        return h

    def __repr__(self):
        return '{}({}x Linear) Dropout(p={})'.format(self.__class__.__name__, self.num_layers, self.dropout)


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.NodeEmb = NodeEmb(opt.vocab_size, opt.d_emb_size, opt.d_emb_size, opt.d_gnn_num_layers, opt.d_gnn_dropout)
        self.GraphEmb = GraphEmb(opt.d_emb_size, opt.d_hidden_size, opt.g_aggr)

    def forward(self, edges, nodes, ns):
        h = self.NodeEmb(edges, nodes)
        h_G = self.GraphEmb(h, ns)
        return h_G


class D(nn.Module):
    def __init__(self, opt):
        super(D, self).__init__()
        self.conv_discriminator = Discriminator(opt)
        self.reduc_discriminator = Discriminator(opt)
        self.label_embedding = nn.Embedding(opt.n_perf_classes, opt.cls_perf_embedding_size)
        self.GetAcc = GetAcc(opt.d_hidden_size*2+opt.cls_perf_embedding_size, opt.d_mlp_num_layers, opt.d_mlp_dropout)

    def forward(self, conv_edges, conv_nodes, conv_ns, reduc_edges, reduc_nodes, reduc_ns, labels):
        conv_h_G = self.conv_discriminator(conv_edges, conv_nodes, conv_ns)
        reduc_h_G = self.reduc_discriminator(reduc_edges, reduc_nodes, reduc_ns)
        acc = self.GetAcc(torch.cat([conv_h_G, reduc_h_G, F.normalize(self.label_embedding(labels), dim=-1)], -1))
        return acc


class D_MultiObj(nn.Module):
    def __init__(self, opt):
        super(D_MultiObj, self).__init__()
        self.conv_discriminator = Discriminator(opt)
        self.reduc_discriminator = Discriminator(opt)
        self.perf_label_embedding = nn.Embedding(opt.n_perf_classes, opt.cls_perf_embedding_size)
        self.param_label_embedding = nn.Embedding(opt.n_params_classes, opt.cls_params_embedding_size)
        self.GetAcc = GetAcc(opt.d_hidden_size*2+opt.cls_perf_embedding_size+opt.cls_params_embedding_size, opt.d_mlp_num_layers, opt.d_mlp_dropout)

    def forward(self, conv_edges, conv_nodes, conv_ns, reduc_edges, reduc_nodes, reduc_ns, perf_labels, param_labels):
        conv_h_G = self.conv_discriminator(conv_edges, conv_nodes, conv_ns)
        reduc_h_G = self.reduc_discriminator(reduc_edges, reduc_nodes, reduc_ns)
        acc = self.GetAcc(torch.cat([conv_h_G, reduc_h_G, F.normalize(self.perf_label_embedding(perf_labels), dim=-1), F.normalize(self.param_label_embedding(param_labels), dim=-1)], -1))
        return acc
