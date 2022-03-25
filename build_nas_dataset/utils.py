import os
import copy
import numpy as np
import logging
import shutil
import threading
import gc
import zipfile
from io import BytesIO
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

INPUT0 = 'input0'
INPUT1 = 'input1'
OUTPUT = 'output'
SEQCONV3X3 = 'seqconv3x3'
SEQCONV5X5 = 'seqconv5x5'
AVGPOOL3X3 = 'avgpool3x3'
MAXPOOL3X3 = 'maxpool3x3'
IDENTITY = 'identity'
CODE = {OUTPUT: 0, INPUT0: 1, INPUT1: 2, SEQCONV3X3: 3, SEQCONV5X5: 4, AVGPOOL3X3: 5, MAXPOOL3X3: 6, IDENTITY: 7}
CODE_ = {0: OUTPUT, 1: INPUT0, 2: INPUT1, 3: SEQCONV3X3, 4: SEQCONV5X5, 5: AVGPOOL3X3, 6: MAXPOOL3X3, 7: IDENTITY}

def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt
      

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
    
    
def _data_transforms_cifar10(cutout_size):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if cutout_size is not None:
        train_transform.transforms.append(Cutout(cutout_size))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    return train_transform, valid_transform


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def convert_to_pil(bytes_obj):
    img = Image.open(BytesIO(bytes_obj))
    return img.convert('RGB')


class ReadImageThread(threading.Thread):
    def __init__(self, root, fnames, class_id, target_list):
        threading.Thread.__init__(self)
        self.root = root
        self.fnames = fnames
        self.class_id = class_id
        self.target_list = target_list
        
    def run(self):
        for fname in self.fnames:
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                path = os.path.join(self.root, fname)
                with open(path, 'rb') as f:
                    image = f.read()
                item = (image, self.class_id)
                self.target_list.append(item)


class InMemoryDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryDataset, self).__init__()
        self.path = path
        self.transform = transform
        self.samples = []
        classes, class_to_idx = self.find_classes(self.path)
        dir = os.path.expanduser(self.path)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                if num_workers == 1:
                    for fname in sorted(fnames):
                        if has_file_allowed_extension(fname, IMG_EXTENSIONS):
                            path = os.path.join(root, fname)
                            with open(path, 'rb') as f:
                                image = f.read()
                            item = (image, class_to_idx[target])
                            self.samples.append(item)
                else:
                    fnames = sorted(fnames)
                    num_files = len(fnames)
                    threads = []
                    res = [[] for i in range(num_workers)]
                    num_per_worker = num_files // num_workers
                    for i in range(num_workers):
                        start_index = num_per_worker * i
                        end_index = num_files if i == num_workers - 1 else num_per_worker * (i+1)
                        thread = ReadImageThread(root, fnames[start_index:end_index], class_to_idx[target], res[i])
                        threads.append(thread)
                    for thread in threads:
                        thread.start()
                    for thread in threads:
                        thread.join()
                    for item in res:
                        self.samples += item
                    del res, threads
                    gc.collect()
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    @staticmethod
    def find_classes(root):
        classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx
    

class ZipDataset(data.Dataset):
    def __init__(self, path, transform=None):
        super(ZipDataset, self).__init__()
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.samples = []
        with zipfile.ZipFile(self.path, 'r') as reader:
            classes, class_to_idx = self.find_classes(reader)
            fnames = sorted(reader.namelist())
        for fname in fnames:
            if self.is_directory(fname):
                continue
            target = self.get_target(fname)
            item = (fname, class_to_idx[target])
            self.samples.append(item)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        with zipfile.ZipFile(self.path, 'r') as reader:
            sample = reader.read(sample)
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    @staticmethod
    def is_directory(fname):
        if fname.startswith('n') and fname.endswith('/'):
            return True
        return False
    
    @staticmethod
    def get_target(fname):
        assert fname.startswith('n')
        return fname.split('/')[0]
    
    @staticmethod
    def find_classes(reader):
        classes = [ZipDataset.get_target(name) for name in reader.namelist() if ZipDataset.is_directory(name)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


class ReadZipImageThread(threading.Thread):
    def __init__(self, reader, fnames, class_to_idx, target_list):
        threading.Thread.__init__(self)
        self.reader = reader
        self.fnames = fnames
        self.target_list = target_list
        self.class_to_idx = class_to_idx
    
    def run(self):
        for fname in self.fnames:
            if InMemoryZipDataset.is_directory(fname):
                continue
            image = self.reader.read(fname)
            class_id = self.class_to_idx[InMemoryZipDataset.get_target(fname)]
            item = (image, class_id)
            self.target_list.append(item)


class InMemoryZipDataset(data.Dataset):
    def __init__(self, path, transform=None, num_workers=1):
        super(InMemoryZipDataset, self).__init__()
        self.path = os.path.expanduser(path)
        self.transform = transform
        self.samples = []
        reader = zipfile.ZipFile(self.path, 'r')
        classes, class_to_idx = self.find_classes(reader)
        fnames = sorted(reader.namelist())
        if num_workers == 1:
            for fname in fnames:
                if self.is_directory(fname):
                    continue
                target = self.get_target(fname)
                image = reader.read(fname)
                item = (image, class_to_idx[target])
                self.samples.append(item)
        else:
            num_files = len(fnames)
            threads = []
            res = [[] for i in range(num_workers)]
            num_per_worker = num_files // num_workers
            for i in range(num_workers):
                start_index = num_per_worker * i
                end_index = num_files if i == num_workers - 1 else (i+1) * num_per_worker
                thread = ReadZipImageThread(reader, fnames[start_index:end_index], class_to_idx, res[i])
                threads.append(thread)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            for item in res:
                self.samples += item
            del res, threads
            gc.collect()
        reader.close()
            
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample, target = self.samples[index]
        sample = convert_to_pil(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
    
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.path)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
    
    @staticmethod
    def is_directory(fname):
        if fname.startswith('n') and fname.endswith('/'):
            return True
        return False
    
    @staticmethod
    def get_target(fname):
        assert fname.startswith('n')
        return fname.split('/')[0]

    @staticmethod
    def find_classes(fname):
        classes = [ZipDataset.get_target(name) for name in fname.namelist() if ZipDataset.is_directory(name)]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)
      

def save(model_path, args, model, epoch, step, optimizer, best_acc_top1, is_best=True):
    if hasattr(model, 'module'):
        model = model.module
    state_dict = {
        'args': args,
        'model': model.state_dict() if model else {},
        'epoch': epoch,
        'step': step,
        'optimizer': optimizer.state_dict(),
        'best_acc_top1': best_acc_top1,
    }
    filename = os.path.join(model_path, 'checkpoint{}.pt'.format(epoch))
    torch.save(state_dict, filename)
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    shutil.copyfile(filename, newest_filename)
    if is_best:
        best_filename = os.path.join(model_path, 'checkpoint_best.pt')
        shutil.copyfile(filename, best_filename)
  

def load(model_path):
    newest_filename = os.path.join(model_path, 'checkpoint.pt')
    if not os.path.exists(newest_filename):
        return None, None, 0, 0, None, 0
    state_dict = torch.load(newest_filename)
    args = state_dict['args']
    model_state_dict = state_dict['model']
    epoch = state_dict['epoch']
    step = state_dict['step']
    optimizer_state_dict = state_dict['optimizer']
    best_acc_top1 = state_dict.get('best_acc_top1')
    return args, model_state_dict, epoch, step, optimizer_state_dict, best_acc_top1


def sample_arch(arch_pool, prob=None):
    N = len(arch_pool)
    indices = [i for i in range(N)]
    if prob is not None:
        prob = np.array(prob, dtype=np.float32)
        prob = prob / prob.sum()
        index = np.random.choice(indices, p=prob)
    else:
        index = np.random.choice(indices)
    arch = arch_pool[index]
    return arch


def generate_arch(n, max_num_node):
  prob = np.array([1] + [2**i for i in range(3, max_num_node)], dtype=np.float32)
  prob = prob / prob.sum()
  num_node_list = [i for i in range(3, max_num_node+1)]

  def _get_arch():
    num_node = np.random.choice(num_node_list, p=prob)
    bias = np.random.random()
    edge = np.triu((np.random.random([num_node, num_node])+bias).astype(int), 1)
    edge[:2,:2] = 0
    edge = edge.tolist()
    node = np.random.randint(3, len(CODE), [num_node-3])
    node = [INPUT0] + [INPUT1] + [CODE_[x] for x in node] + [OUTPUT]
    return [edge, node]

  pruned_archs = []
  while len(pruned_archs)<n:
    need = n-len(pruned_archs)
    archs = [[_get_arch(), _get_arch()] for i in range(need)]
    for arch in archs:
      conv, reduc = arch[0], arch[1]
      conv = prune(conv)
      reduc = prune(reduc)
      if conv is not None and reduc is not None and [conv, reduc] not in pruned_archs:
        pruned_archs.append([conv, reduc])
  return pruned_archs #[[[conv_edge, conv_node],[reduc_edge, reduc_node]], ...]


def prune(arch):
  """Prune the extraneous parts of the graph.

  General procedure:
    1) Remove parts of graph not connected to input.
    2) Remove parts of graph not connected to output.
    3) Reorder the vertices so that they are consecutive after steps 1 and 2.

  These 3 steps can be combined by deleting the rows and columns of the
  vertices that are not reachable from both the input and output (in reverse).
  """
  edge, node = arch[0], arch[1]
  if not isinstance(edge, np.ndarray):
    edge = np.array(edge)
  shape = np.shape(edge)
  if len(shape) != 2 or shape[0] != shape[1]:
    raise ValueError('edge must be square')
  if shape[0] != len(node):
    raise ValueError('length of node must match edge dimensions')
  if not is_upper_triangular(edge):
    raise ValueError('edge must be upper triangular')

  original_edge = copy.deepcopy(edge)
  original_node = copy.deepcopy(node)

  edge = copy.deepcopy(edge)
  node = copy.deepcopy(node)

  num_vertices = np.shape(original_edge)[0]

  # DFS forward from input
  visited_from_input = set([0, 1])
  frontier = [0, 1]
  while frontier:
    top = frontier.pop()
    for v in range(top + 1, num_vertices):
      if original_edge[top, v] and v not in visited_from_input:
        visited_from_input.add(v)
        frontier.append(v)

  # DFS backward from output
  visited_from_output = set([num_vertices - 1])
  frontier = [num_vertices - 1]
  while frontier:
    top = frontier.pop()
    for v in range(0, top):
      if original_edge[v, top] and v not in visited_from_output:
        visited_from_output.add(v)
        frontier.append(v)

  # Any vertex that isn't connected to both input and output is extraneous to
  # the computation graph.
  extraneous = set(range(num_vertices)).difference(
      visited_from_input.intersection(visited_from_output))
  # If the non-extraneous graph is less than 3 vertices, the input is not
  # connected to the output and the spec is invalid.
  if len(extraneous) > num_vertices - 3:
    edge = None
    node = None
    return

  edge = np.delete(edge, list(extraneous), axis=0)
  edge = np.delete(edge, list(extraneous), axis=1)
  for index in sorted(extraneous, reverse=True):
    del node[index]
  return [edge.tolist(), node]


def is_upper_triangular(matrix):
  """True if matrix is 0 on diagonal and below."""
  for src in range(np.shape(matrix)[0]):
    for dst in range(0, src + 1):
      if matrix[src, dst] != 0:
        return False
  return True


def calculate_A_to_DAD(A):
  degrees = torch.pow(A.sum(1).float(), -0.5)
  D = torch.diag(degrees)
  return torch.matmul(torch.matmul(A, D).t(), D)


def normalize_target(target_list):
  min_val = min(target_list)
  max_val = max(target_list)
  res = [(i - min_val) / (max_val - min_val) for i in target_list]
  return res


def generate_eval_points(eval_epochs, stand_alone_epoch, total_epochs):
    if isinstance(eval_epochs, list):
        return eval_epochs
    assert isinstance(eval_epochs, int)
    res = []
    eval_point = eval_epochs - stand_alone_epoch
    while eval_point + stand_alone_epoch <= total_epochs:
        res.append(eval_point)
        eval_point += eval_epochs
    return res


class ModelSpec(object):
  """Model specification given adjacency matrix and labeling."""

  def __init__(self, matrix, ops, data_format='channels_last'):
    """Initialize the module spec.

    Args:
      matrix: ndarray or nested list with shape [V, V] for the adjacency matrix.
      ops: V-length list of labels for the base ops used. The first and last
        elements are ignored because they are the input and output vertices
        which have no operations. The elements are retained to keep consistent
        indexing.
      data_format: channels_last or channels_first.

    Raises:
      ValueError: invalid matrix or ops
    """
    if not isinstance(matrix, np.ndarray):
      matrix = np.array(matrix)
    shape = np.shape(matrix)
    if len(shape) != 2 or shape[0] != shape[1]:
      raise ValueError('matrix must be square')
    if shape[0] != len(ops):
      raise ValueError('length of ops must match matrix dimensions')
    if not is_upper_triangular(matrix):
      raise ValueError('matrix must be upper triangular')

    # Both the original and pruned matrices are deep copies of the matrix and
    # ops so any changes to those after initialization are not recognized by the
    # spec.
    self.original_matrix = copy.deepcopy(matrix)
    self.original_ops = copy.deepcopy(ops)

    self.matrix = copy.deepcopy(matrix)
    self.ops = copy.deepcopy(ops)
    self.valid_spec = True
    self._prune()

    self.data_format = data_format

  def _prune(self):
    """Prune the extraneous parts of the graph.

    General procedure:
      1) Remove parts of graph not connected to input.
      2) Remove parts of graph not connected to output.
      3) Reorder the vertices so that they are consecutive after steps 1 and 2.

    These 3 steps can be combined by deleting the rows and columns of the
    vertices that are not reachable from both the input and output (in reverse).
    """
    num_vertices = np.shape(self.original_matrix)[0]

    # DFS forward from input
    visited_from_input = set([0, 1])
    frontier = [0, 1]
    while frontier:
      top = frontier.pop()
      for v in range(top + 1, num_vertices):
        if self.original_matrix[top, v] and v not in visited_from_input:
          visited_from_input.add(v)
          frontier.append(v)

    # DFS backward from output
    visited_from_output = set([num_vertices - 1])
    frontier = [num_vertices - 1]
    while frontier:
      top = frontier.pop()
      for v in range(0, top):
        if self.original_matrix[v, top] and v not in visited_from_output:
          visited_from_output.add(v)
          frontier.append(v)

    # Any vertex that isn't connected to both input and output is extraneous to
    # the computation graph.
    extraneous = set(range(num_vertices)).difference(
        visited_from_input.intersection(visited_from_output))

    # If the non-extraneous graph is less than 3 vertices, the input is not
    # connected to the output and the spec is invalid.
    if len(extraneous) > num_vertices - 3:
      self.matrix = None
      self.ops = None
      self.valid_spec = False
      return

    self.matrix = np.delete(self.matrix, list(extraneous), axis=0)
    self.matrix = np.delete(self.matrix, list(extraneous), axis=1)
    for index in sorted(extraneous, reverse=True):
      del self.ops[index]

  def hash_spec(self, canonical_ops):
    """Computes the isomorphism-invariant graph hash of this spec.

    Args:
      canonical_ops: list of operations in the canonical ordering which they
        were assigned (i.e. the order provided in the config['available_ops']).

    Returns:
      MD5 hash of this spec which can be used to query the dataset.
    """
    # Invert the operations back to integer label indices used in graph gen.
    labeling = [-1] + [canonical_ops.index(op) for op in self.ops[1:-1]] + [-2]
    return graph_util.hash_module(self.matrix, labeling)

  def visualize(self):
    """Creates a dot graph. Can be visualized in colab directly."""
    num_vertices = np.shape(self.matrix)[0]
    g = graphviz.Digraph()
    g.node(str(0), 'input0')
    g.node(str(1), 'input1')
    for v in range(2, num_vertices - 1):
      g.node(str(v), self.ops[v])
    g.node(str(num_vertices - 1), 'output')

    for src in range(num_vertices - 1):
      for dst in range(src + 1, num_vertices):
        if self.matrix[src, dst]:
          g.edge(str(src), str(dst))

    return g


def is_upper_triangular(matrix):
  """True if matrix is 0 on diagonal and below."""
  for src in range(np.shape(matrix)[0]):
    for dst in range(0, src + 1):
      if matrix[src, dst] != 0:
        return False

  return True
