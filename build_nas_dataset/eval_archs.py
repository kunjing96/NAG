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

import sys, os, random, logging, argparse, gc
import numpy as np
import utils

# Basic model parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/cifar10')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100'])
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--batch_size', type=int, default=128)
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
    spec = kwargs.pop('spec')

    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)

    conv_spec = spec[0]#_ToModelSpec(spec[0][0], spec[0][1])
    reduc_spec = spec[1]#_ToModelSpec(spec[1][0], spec[1][1])
    model = NASNetworkCIFAR(args, 10, args.layers, args.channels, args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.steps, [conv_spec, reduc_spec])
    #logging.info(model)
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


'''def build_cifar100(model_state_dict=None, optimizer_state_dict=None, **kwargs):
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
    model = NASNetworkCIFAR(args, 10, args.layers, args.channels, args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.steps, [conv_spec, reduc_spec])
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
        weight_decay=args.child_l2_reg,
    )
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler'''


'''def build_imagenet(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    ratio = kwargs.pop('ratio')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    valid_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    if args.zip_file:
        logging.info('Loading data from zip file')
        traindir = os.path.join(args.data, 'train.zip')
        if args.lazy_load:
            train_data = utils.ZipDataset(traindir, train_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryZipDataset(traindir, train_transform, num_workers=32)
    else:
        logging.info('Loading data from directory')
        traindir = os.path.join(args.data, 'train')
        if args.lazy_load:
            train_data = dset.ImageFolder(traindir, train_transform)
        else:
            logging.info('Loading data into memory')
            train_data = utils.InMemoryDataset(traindir, train_transform, num_workers=32)
       
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(ratio * num_train))
    train_indices = sorted(indices[:split])
    valid_indices = sorted(indices[split:])

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(valid_indices),
        pin_memory=True, num_workers=16)
    
    model = NASWSNetworkImageNet(1000, args.layers, args.max_num_vertices, args.channels, args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = CrossEntropyLabelSmooth(1000, args.label_smooth).cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler'''


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

    N, CLS = 50, 2
    acc_list = []
    spec_list = torch.load('../archs_{}'.format(CLS))[:N]
    logging.info('Get {} archs...'.format(len(spec_list)))
    for i, spec in enumerate(spec_list):
        logging.info('{}th arch...'.format(i))
        if spec[0].ops is None or spec[0].matrix is None or spec[1].ops is None or spec[1].matrix is None:
            acc_list.append(-1)
            continue
        _, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1 = utils.load(args.output_dir)
        build_fn = get_builder(args.dataset)
        train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(model_state_dict, optimizer_state_dict, epoch=epoch-1, spec=spec)

        while epoch < 50:
            scheduler.step()
            logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
            train_acc, train_obj, step = train(train_queue, model, optimizer, step, train_criterion)
            logging.info('train_acc %f', train_acc)
            valid_acc_top1, valid_obj = valid(valid_queue, model, eval_criterion)
            logging.info('valid_acc %f', valid_acc_top1)
            epoch += 1
        acc_list.append(valid_acc_top1)
        logging.info('conv_spec')
        logging.info('matrix\n{}'.format(spec[0].matrix))
        logging.info('ops\n{}'.format(spec[0].ops))
        logging.info('reduc_spec')
        logging.info('matrix\n{}'.format(spec[1].matrix))
        logging.info('ops\n{}'.format(spec[1].ops))
        logging.info('valid_acc %f', valid_acc_top1)
        logging.info('---------------------------------')
        del train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler
        gc.collect()

    logging.info('The best arch is {}.'.format(np.argmax(acc_list)))


if __name__ == '__main__':
    main()
