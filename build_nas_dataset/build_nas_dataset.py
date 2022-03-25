import sys, os, time, copy, random, logging, argparse
import numpy as np
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms

import utils
from model import NASNetworkCIFAR
from model_search import NASWSNetworkCIFAR
from model_spec import ModelSpec

parser = argparse.ArgumentParser(description='GANAS CIFAR-10')
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100, imagenet'])
parser.add_argument('--zip_file', action='store_true', default=False)
parser.add_argument('--lazy_load', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--batch_size', default=128, type=int, help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--epochs', default=150, type=int, help='#epochs of training')
parser.add_argument('--layers', default=3, type=int, help='#layers per stack')
parser.add_argument('--max_num_vertices', default=10, type=int, help='#max nodes')
parser.add_argument('--channels', default=20, type=int, help='output channels of stem convolution')
parser.add_argument('--cutout_size', type=int, default=16)
parser.add_argument('--grad_clip', default=5.0, type=float, help='gradient clipping')
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--lr_min', type=float, default=0.001)
parser.add_argument('--keep_prob', type=float, default=1.0)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.9)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--use_aux_head', action='store_false', default=True)
parser.add_argument('--lr', default=0.1, type=float, help='base learning rate')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--num_labels', default=10, type=int, help='#classes')
parser.add_argument('--available_ops', default=['seqconv3x3', 'seqconv5x5', 'avgpool3x3', 'maxpool3x3', 'identity'], type=list, help='available operations performed on vertex')
parser.add_argument('--num_sampled_arch', type=int, default=4000)
parser.add_argument('--arch_pool', type=str, default=None)
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

os.makedirs("sampled_nasdata", exist_ok=True)

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def get_builder(dataset):
    if dataset == 'cifar10':
        return build_cifar10
    elif dataset == 'cifar100':
        return build_cifar100
    else:
        return build_imagenet
    

def build_cifar10(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=valid_transform)
    
    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train)) 
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)
    
    model = NASWSNetworkCIFAR(args, 10, args.layers, args.max_num_vertices, args.channels, args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min, epoch)
    return train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler


'''def build_cifar100(model_state_dict=None, optimizer_state_dict=None, **kwargs):
    epoch = kwargs.pop('epoch')
    ratio = kwargs.pop('ratio')
    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR100(root=args.data, train=True, download=True, transform=valid_transform)

    num_train = len(train_data)
    assert num_train == len(valid_data)
    indices = list(range(num_train))    
    split = int(np.floor(ratio * num_train))
    np.random.shuffle(indices)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        pin_memory=True, num_workers=16)
    
    model = NASWSNetworkCIFAR(100, args.layers, args.max_num_vertices, args.channels, args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.steps)
    model = model.cuda()
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.lr_max,
        momentum=0.9,
        weight_decay=args.child_l2_reg,
    )
    if model_state_dict is not None:
        model.load_state_dict(model_state_dict)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, args.lr_min, epoch)
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


def child_train(train_queue, model, optimizer, global_step, spec_pool, spec_pool_prob, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        optimizer.zero_grad()
        # sample an spec to train
        spec = utils.sample_arch(spec_pool, spec_pool_prob)
        logits, aux_logits = model(input, spec, global_step)
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


def child_valid(valid_queue, model, spec_pool, criterion):
    valid_acc_list = []
    with torch.no_grad():
        model.eval()
        for i, spec in enumerate(spec_pool):
            inputs, targets = next(iter(valid_queue))
            inputs = inputs.cuda()
            targets = targets.cuda()

            logits, _ = model(inputs, spec, bn_train=True)
            loss = criterion(logits, targets)

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            valid_acc_list.append(prec1)

            if (i+1) % 100 == 0:
                logging.info('Valid loss %.2f top1 %f top5 %f', loss, prec1, prec5)

    return valid_acc_list


def train_and_evaluate_top_on_cifar10(specs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, spec in enumerate(specs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} spec'.format(i+1))
        model = NASNetworkCIFAR(args, 10, args.layers, args.channels, 0.6, 0.8, True, args.steps, spec)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr_max,
            momentum=0.9,
            weight_decay=args.l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, args.lr_min)
        global_step = 0
        for e in range(10):
            scheduler.step()
            for step, (input, target) in enumerate(train_queue):
                input = input.cuda().requires_grad_()
                target = target.cuda()

                optimizer.zero_grad()
                # sample an spec to train
                logits, aux_logits = model(input, global_step)
                global_step += 1
                loss = train_criterion(logits, target)
                if aux_logits is not None:
                    aux_loss = train_criterion(aux_logits, target)
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
                    logging.info('Train epoch %03d %03d loss %e top1 %f top5 %f', e+1, step+1, objs.avg, top1.avg, top5.avg)
        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)

                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg)
    return res


'''def train_and_evaluate_top_on_cifar100(specs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, spec in enumerate(specs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} spec'.format(i+1))
        model = NASNetworkCIFAR(args, 100, args.child_layers, args.child_nodes, args.child_channels, 0.6, 0.8, True, args.steps, spec)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr_max,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10, args.child_lr_min)
        global_step = 0
        for e in range(10):
            scheduler.step()
            for step, (input, target) in enumerate(train_queue):
                input = input.cuda().requires_grad_()
                target = target.cuda()

                optimizer.zero_grad()
                # sample an spec to train
                logits, aux_logits = model(input, global_step)
                global_step += 1
                loss = train_criterion(logits, target)
                if aux_logits is not None:
                    aux_loss = train_criterion(aux_logits, target)
                    loss += 0.4 * aux_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
                optimizer.step()

                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('Train %3d %03d loss %e top1 %f top5 %f', e+1, step+1, objs.avg, top1.avg, top5.avg)
        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg)
    return res'''


'''def train_and_evaluate_top_on_imagenet(specs, train_queue, valid_queue):
    res = []
    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    for i, spec in enumerate(specs):
        objs.reset()
        top1.reset()
        top5.reset()
        logging.info('Train and evaluate the {} spec'.format(i+1))
        model = NASNetworkImageNet(args, 1000, args.child_layers, args.child_nodes, args.child_channels, 1.0, 1.0, True, args.steps, spec)
        model = model.cuda()
        model.train()
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.child_lr,
            momentum=0.9,
            weight_decay=args.child_l2_reg,
        )
        for step, (input, target) in enumerate(train_queue):
            input = input.cuda().requires_grad_()
            target = target.cuda()

            optimizer.zero_grad()
            # sample an spec to train
            logits, aux_logits = model(input, step)
            loss = train_criterion(logits, target)
            if aux_logits is not None:
                aux_loss = train_criterion(aux_logits, target)
                loss += 0.4 * aux_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.child_grad_bound)
            optimizer.step()
            
            prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
            
            if (step+1) % 100 == 0:
                logging.info('Train %03d loss %e top1 %f top5 %f', step+1, objs.avg, top1.avg, top5.avg)
            if step+1 == 500:
                break

        objs.reset()
        top1.reset()
        top5.reset()
        with torch.no_grad():
            model.eval()
            for step, (input, target) in enumerate(valid_queue):
                input = input.cuda()
                target = target.cuda()
            
                logits, _ = model(input)
                loss = eval_criterion(logits, target)
            
                prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
                n = input.size(0)
                objs.update(loss.data, n)
                top1.update(prec1.data, n)
                top5.update(prec5.data, n)
            
                if (step+1) % 100 == 0:
                    logging.info('valid %03d %e %f %f', step+1, objs.avg, top1.avg, top5.avg)
        res.append(top1.avg)
    return res'''


def _ToModelSpec(mat, ops):
    return ModelSpec(mat, ops)


def SaveAsDataset(spec_pool, perf_list, param_size_list, data_file):
    data = []
    for i, spec in enumerate(spec_pool):
        conv_spec, reduc_spec = spec[0], spec[1]
        conv_edge = torch.FloatTensor(conv_spec.matrix.T)
        reduc_edge = torch.FloatTensor(reduc_spec.matrix.T)
        conv_node = torch.LongTensor([utils.CODE[x] for x in conv_spec.ops]).unsqueeze(-1)
        reduc_node = torch.LongTensor([utils.CODE[x] for x in reduc_spec.ops]).unsqueeze(-1)
        conv_node = torch.zeros([conv_node.size(0), len(utils.CODE)], dtype=torch.float).scatter(1, conv_node, 1)
        reduc_node = torch.zeros([reduc_node.size(0), len(utils.CODE)], dtype=torch.float).scatter(1, reduc_node, 1)
        perf = torch.FloatTensor([perf_list[i]])
        params = torch.LongTensor([param_size_list[i]])
        conv_n = torch.LongTensor([len(conv_spec.ops)])
        reduc_n = torch.LongTensor([len(reduc_spec.ops)])
        data.append({'conv_edge': conv_edge, 'conv_node': conv_node, 'reduc_edge': reduc_edge, 'reduc_node': reduc_node, 'perf': perf, 'conv_n': conv_n, 'reduc_n': reduc_n, 'params': params})
    torch.save(data, data_file)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
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

    args.steps = int(np.ceil(45000 / args.batch_size)) * args.epochs

    logging.info("args = %s", args)

    # Build WS-Model
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model, train_criterion, eval_criterion, optimizer, scheduler = build_fn(ratio=0.9, epoch=-1)

    # Sample spec pool
    spec_pool = utils.generate_arch(args.num_sampled_arch, args.max_num_vertices)
    logging.info("Successfully sample {} archs.".format(len(spec_pool)))
    spec_pool = [[_ToModelSpec(spec[0][0], spec[0][1]), _ToModelSpec(spec[0][0], spec[0][1])] for spec in spec_pool]
    arch_pool_prob = None

    param_size_list = [model.get_param_size(spec) for spec in spec_pool]

    history_acc_list_list = []
    # Train child model
    step = 0
    for epoch in range(1, args.epochs + 1):
        scheduler.step()
        lr = scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)
        # sample a spec to train
        train_acc, train_obj, step = child_train(train_queue, model, optimizer, step, spec_pool, arch_pool_prob, train_criterion)
        logging.info('train_acc %f', train_acc)
        # Evaluate specs in spec pool
        history_acc_list_list.append(child_valid(valid_queue, model, spec_pool, eval_criterion))
        valid_accuracy_list = np.mean(history_acc_list_list[-10:], 0)
        if epoch%5 == 0:
            # Get dataset
            SaveAsDataset(spec_pool, valid_accuracy_list, param_size_list, './sampled_nasdata/nasdata_{}'.format(epoch))

if __name__ == '__main__':
    main()
