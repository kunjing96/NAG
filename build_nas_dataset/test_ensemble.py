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

# Basic model parameters.
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/cifar10')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10, cifar100'])
parser.add_argument('--output_dir_list', type=str, default=['../model14el/models/', '../model24el/models/', '../model34el/models/'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=600)
parser.add_argument('--layers', type=int, default=6)
parser.add_argument('--max_num_vertices', type=int, default=10)
parser.add_argument('--channels', type=int, default=36)
parser.add_argument('--cutout_size', type=int, default=None)
parser.add_argument('--grad_clip', type=float, default=5.0)
parser.add_argument('--lr_max', type=float, default=0.025)
parser.add_argument('--lr_min', type=float, default=0)
parser.add_argument('--keep_prob', type=float, default=0.6)
parser.add_argument('--drop_path_keep_prob', type=float, default=0.8)
parser.add_argument('--l2_reg', type=float, default=3e-4)
parser.add_argument('--use_aux_head', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=0)
args = parser.parse_args()


spec_list = [
[
([[0, 0, 1, 1, 1, 0, 1, 0],
  [0, 0, 1, 1, 1, 1, 0, 0],
  [0, 0, 0, 1, 0, 1, 1, 0],
  [0, 0, 0, 0, 1, 0, 1, 1],
  [0, 0, 0, 0, 0, 0, 1, 1],
  [0, 0, 0, 0, 0, 0, 1, 0],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 0]],
  ['input0', 'input1', 'seqconv3x3', 'avgpool3x3', 'maxpool3x3', 'maxpool3x3', 'maxpool3x3', 'output']),
([[0, 0, 1, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  [0, 0, 0, 0]],
  ['input0', 'input1', 'maxpool3x3', 'output'])
],
[
([[0, 0, 1, 1, 1],
  [0, 0, 1, 0, 0],
  [0, 0, 0, 1, 0],
  [0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0]],
  ['input0', 'input1', 'seqconv3x3', 'avgpool3x3', 'output']),
([[0, 0, 1, 1],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
  [0, 0, 0, 0]],
  ['input0', 'input1', 'seqconv3x3', 'output'])
],
[
([[0, 0, 1, 0, 0, 1, 0, 0],
  [0, 0, 1, 1, 1, 1, 1, 1],
  [0, 0, 0, 1, 1, 1, 1, 1],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 1, 0, 0],
  [0, 0, 0, 0, 0, 0, 1, 1],
  [0, 0, 0, 0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0, 0, 0, 0]],
  ['input0', 'input1', 'maxpool3x3', 'identity', 'seqconv5x5', 'maxpool3x3', 'seqconv5x5', 'output']),
([[0, 0, 1, 1, 1],
  [0, 0, 1, 1, 0],
  [0, 0, 0, 1, 1],
  [0, 0, 0, 0, 1],
  [0, 0, 0, 0, 0]],
  ['input0', 'input1', 'maxpool3x3', 'maxpool3x3', 'output'])
]
]

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')


def _ToModelSpec(mat, ops):
    return ModelSpec(mat, ops)


def valid(valid_queue, model_list, voting):
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda()

            logits_list = []
            for model in model_list:
                model.eval()
                logits, _ = model(input)
                logits_list.append(logits)

            prec1, prec5 = ensemble_accuracy(logits_list, target, topk=(1, 5), voting=voting)
            n = input.size(0)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
        
            if (step+1) % 100 == 0:
                logging.info('Valid %03d %f %f', step+1, top1.avg, top5.avg)

    return top1.avg


def ensemble_accuracy(output_list, target, topk=(1,), voting='hard'):
    maxk = max(topk)
    batch_size = target.size(0)

    if voting == 'hard':
        pred_list = []
        for output in output_list:
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            pred_list.append(pred)
        #vote
        vote_pred = sum([torch.zeros(maxk, output.size(0), output.size(1)).cuda().scatter(-1, pred.unsqueeze(-1), 1) for pred in pred_list]).long().max(-1)[1].squeeze()
        correct = vote_pred.eq(target.view(1, -1).expand_as(pred))
    elif voting == 'soft':
        output = sum(output_list)/float(len(output_list))
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    else:
        raise ValueError('Voting is excepted as \'hard\' or \'soft\'.')

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


def get_builder(dataset):
    if dataset == 'cifar10':
        return build_cifar10
    elif dataset == 'cifar100':
        return build_cifar100
    else:
        return build_imagenet
    

def build_cifar10(model_state_dict_list):

    train_transform, valid_transform = utils._data_transforms_cifar10(args.cutout_size)
    train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
    valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=16)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.eval_batch_size, shuffle=False, pin_memory=True, num_workers=16)

    model_list = []
    for i, model_state_dict in enumerate(model_state_dict_list):
        conv_spec = _ToModelSpec(spec_list[i][0][0], spec_list[i][0][1])
        reduc_spec = _ToModelSpec(spec_list[i][1][0], spec_list[i][1][1])
        model = NASNetworkCIFAR(args, 10, args.layers, args.channels, args.keep_prob, args.drop_path_keep_prob, args.use_aux_head, args.steps, [conv_spec, reduc_spec])
        logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
        if model_state_dict is not None:
            model.load_state_dict(model_state_dict)
        model = model.cuda()
        model_list.append(model)

    return train_queue, valid_queue, model_list


def load_models(model_path_list):
    args_list = []
    model_state_dict_list = []
    best_acc_top1_list = []
    for model_path in model_path_list:
        newest_filename = os.path.join(model_path, 'checkpoint_best.pt')
        if not os.path.exists(newest_filename):
            raise Exception('The file does not exist.')
        state_dict = torch.load(newest_filename)
        args_list.append(state_dict['args'])
        model_state_dict_list.append(state_dict['model'])
        best_acc_top1_list.append(state_dict.get('best_acc_top1'))
    return args_list, model_state_dict_list, best_acc_top1_list


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

    _, model_state_dict_list, best_acc_top1_list = load_models(args.output_dir_list)
    build_fn = get_builder(args.dataset)
    train_queue, valid_queue, model_list = build_fn(model_state_dict_list)

    valid_acc_top1 = valid(valid_queue, model_list, voting='hard')
    logging.info('ensemble model valid_acc %f', valid_acc_top1)
    logging.info('model1 valid_acc %f', best_acc_top1_list[0])
    logging.info('model2 valid_acc %f', best_acc_top1_list[1])
    logging.info('model3 valid_acc %f', best_acc_top1_list[2])

if __name__ == '__main__':
    main()
