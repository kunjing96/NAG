import torch
import torch.utils
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

from model import NASNetworkCIFAR
from model_spec import ModelSpec

import sys, os, random, logging, argparse
import numpy as np
import utils
from adabelief_pytorch import AdaBelief

# Basic model parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--batch_size', type=int, default=96)
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--layers', type=int, default=6)
parser.add_argument('--max_num_vertices', type=int, default=10)
parser.add_argument('--channels', type=int, default=36)
parser.add_argument('--cutout_size', type=int, default=16)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--lr_min', type=float, default=0)
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--use_aux_head', action='store_false', default=True)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

spec = [
([[0, 0, 1, 0, 0, 0, 1, 0, 1],
  [0, 0, 1, 1, 1, 1, 0, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 1, 1],
  [0, 0, 0, 0, 1, 1, 1, 0, 1],
  [0, 0, 0, 0, 0, 1, 1, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 0, 0]],
  ['input0', 'input1', 'seqconv3x3', 'seqconv5x5', 'maxpool3x3', 'seqconv5x5', 'seqconv5x5', 'seqconv5x5', 'output']),
([[0, 0, 1, 1, 0, 1],
  [0, 0, 1, 1, 1, 1],
  [0, 0, 0, 1, 0, 1],
  [0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0]],
  ['input0', 'input1', 'seqconv5x5', 'seqconv3x3', 'seqconv5x5', 'output'])
]

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')


def _ToModelSpec(mat, ops):
    return ModelSpec(mat, ops)


def train(train_queue, model, optimizer, global_step, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        optimizer.zero_grad()
        logits, aux_logits = model(input, global_step)
        global_step += 1
        loss = criterion(logits, target)
        if aux_logits is not None:
            aux_loss = criterion(aux_logits, target)
            loss += 0.4 * aux_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        
        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        
        if (step+1) % 100 == 0:
            logging.info('Train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg, global_step


def valid(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()
        
            logits, _ = model(input)
            loss = criterion(logits, target)
        
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
        
            if (step+1) % 100 == 0:
                logging.info('Valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg



def get_builder(dataset):
    if dataset == 'cifar10':
        return build_cifar10
    elif dataset == 'cifar100':
        return build_cifar100
    else:
        return build_imagenet
    

def build_cifar10(model_state_dict, optimizer_state_dict, **kwargs):
    epoch = kwargs.pop('epoch')

    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)

    conv_spec = _ToModelSpec(spec[0][0], spec[0][1])
    reduc_spec = _ToModelSpec(spec[1][0], spec[1][1])
    model = NASNetworkCIFAR(args, 10, args.layers, args.channels, args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.steps, [conv_spec, reduc_spec])
    logging.info(model)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    model = model.cuda()

    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    '''optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )'''
    optimizer = AdaBelief(model.parameters(), lr=1e-3, eps=1e-8, betas=(0.9,0.999), weight_decay=5e-4, weight_decouple=False, rectify=False, fixed_decay=False, amsgrad=False)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def build_cifar100(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')

    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR100(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)

    conv_spec = _ToModelSpec(spec[0][0], spec[0][1])
    reduc_spec = _ToModelSpec(spec[1][0], spec[1][1])
    model = NASNetworkCIFAR(args, 100, args.layers, args.channels, args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.steps, [conv_spec, reduc_spec])
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)

    model = model.cuda()

    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


def main():
    if not torch.cuda.is_available():
        logging.info('No GPU found!')
        sys.exit(1)

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    
    args.steps = int(np.ceil(50000 / args.batch_size)) * args.epochs
    logging.info("Args = %s", args)

    _, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1 = utils.load(args.output_dir)
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(model_state_dict, optimizer_state_dict, epoch=epoch-1)

    while epoch < args.epochs:
        #scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        train_acc, train_obj, step = train(train_queue, model, optimizer, step, train_criterion)
        logging.info('train_acc %f', train_acc)
        valid_acc_top1, valid_obj = valid(valid_queue, model, eval_criterion)
        logging.info('valid_acc %f', valid_acc_top1)
        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        utils.save(args.output_dir, args, model, epoch, step, optimizer, best_acc_top1, is_best)
        

if __name__ == '__main__':
    main()
