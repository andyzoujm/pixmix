# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Main script to launch PixMix training on CIFAR-10/100.

Supports WideResNet, ResNeXt models on CIFAR-10 and CIFAR-100 as well
as evaluation on CIFAR-10-C and CIFAR-100-C.

Example usage:
  `python cifar.py`
"""
from __future__ import print_function

import argparse
import os
import shutil
import time

import pixmix_utils as utils
import numpy as np
from third_party.ResNeXt_DenseNet.models.densenet import densenet
from third_party.ResNeXt_DenseNet.models.resnext import resnext29
from third_party.WideResNet_pytorch.wideresnet import WideResNet

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

parser = argparse.ArgumentParser(
    description='Trains a CIFAR Classifier',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument(
    '--data-path',
    type=str,
    default='./data',
    required=True,
    help='Path to CIFAR and CIFAR-C directories')
parser.add_argument(
    '--mixing-set',
    type=str,
    required=True,
    help='Mixing set directory.')
parser.add_argument(
    '--use_300k',
    action='store_true',
    help='use 300K random images as aug data'
)
parser.add_argument(
    '--model',
    '-m',
    type=str,
    default='wrn',
    choices=['wrn', 'densenet', 'resnext'],
    help='Choose architecture.')
# Optimization options
parser.add_argument(
    '--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument(
    '--learning-rate',
    '-lr',
    type=float,
    default=0.1,
    help='Initial learning rate.')
parser.add_argument(
    '--batch-size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--eval-batch-size', type=int, default=1000)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument(
    '--decay',
    '-wd',
    type=float,
    default=0.0005,
    help='Weight decay (L2 penalty).')
# WRN Architecture options
parser.add_argument(
    '--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=4, type=int, help='Widen factor')
parser.add_argument(
    '--droprate', default=0.3, type=float, help='Dropout probability')
# PixMix options
parser.add_argument(
    '--beta',
    default=3,
    type=int,
    help='Severity of mixing')
parser.add_argument(
    '--k',
    default=4,
    type=int,
    help='Mixing iterations')
parser.add_argument(
    '--aug-severity',
    default=3,
    type=int,
    help='Severity of base augmentation operators')
parser.add_argument(
    '--all-ops',
    '-all',
    action='store_true',
    help='Turn on all augmentation operations (+brightness,contrast,color,sharpness).')
# Checkpointing options
parser.add_argument(
    '--save',
    '-s',
    type=str,
    default='./snapshots',
    help='Folder to save checkpoints.')
parser.add_argument(
    '--resume',
    '-r',
    type=str,
    default='',
    help='Checkpoint path for resume / test.')
parser.add_argument('--evaluate', action='store_true', help='Eval only.')
parser.add_argument(
    '--print-freq',
    type=int,
    default=50,
    help='Training loss print frequency (batches).')
# Acceleration
parser.add_argument(
    '--num-workers',
    type=int,
    default=4,
    help='Number of pre-fetching threads.')

args = parser.parse_args()
print(args)

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

CBAR_CORRUPTIONS = [
    "blue_noise_sample", "brownish_noise", "checkerboard_cutout", 
    "inverse_sparkles", "pinch_and_twirl", "ripple", "circular_motion_blur", 
    "lines", "sparkles", "transverse_chromatic_abberation"]

NUM_CLASSES = 100 if args.dataset == 'cifar100' else 10

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))


def pixmix(orig, mixing_pic, preprocess):
  
  mixings = utils.mixings
  tensorize, normalize = preprocess['tensorize'], preprocess['normalize']
  if np.random.random() < 0.5:
    mixed = tensorize(augment_input(orig))
  else:
    mixed = tensorize(orig)
  
  for _ in range(np.random.randint(args.k + 1)):
    
    if np.random.random() < 0.5:
      aug_image_copy = tensorize(augment_input(orig))
    else:
      aug_image_copy = tensorize(mixing_pic)

    mixed_op = np.random.choice(mixings)
    mixed = mixed_op(mixed, aug_image_copy, args.beta)
    mixed = torch.clip(mixed, 0, 1)

  return normalize(mixed)

def augment_input(image):
  aug_list = utils.augmentations_all if args.all_ops else utils.augmentations
  op = np.random.choice(aug_list)
  return op(image.copy(), args.aug_severity)

class RandomImages300K(torch.utils.data.Dataset):
    def __init__(self, file, transform):
        self.dataset = np.load(file)
        self.transform = transform

    def __getitem__(self, index):
        img = self.dataset[index]
        return self.transform(img), 0

    def __len__(self):
        return len(self.dataset)

class PixMixDataset(torch.utils.data.Dataset):
  """Dataset wrapper to perform PixMix."""

  def __init__(self, dataset, mixing_set, preprocess):
    self.dataset = dataset
    self.mixing_set = mixing_set
    self.preprocess = preprocess

  def __getitem__(self, i):
    x, y = self.dataset[i]
    rnd_idx = np.random.choice(len(self.mixing_set))
    mixing_pic, _ = self.mixing_set[rnd_idx]
    return pixmix(x, mixing_pic, self.preprocess), y

  def __len__(self):
    return len(self.dataset)


def train(net, train_loader, optimizer, scheduler):
  """Train for one epoch."""
  net.train()
  loss_ema = 0.
  for i, (images, targets) in enumerate(train_loader):

    optimizer.zero_grad()

    images = images.cuda()
    targets = targets.cuda()
    logits = net(images)
    loss = F.cross_entropy(logits, targets)
    loss.backward()
    optimizer.step()
    scheduler.step()
    loss_ema = loss_ema * 0.9 + float(loss) * 0.1
    # if i % args.print_freq == 0:
    #   print('Train Loss {:.3f}'.format(loss_ema))

  return loss_ema


def test(net, test_loader, adv=None):
  """Evaluate network on given dataset."""
  net.eval()
  total_loss = 0.
  total_correct = 0
  with torch.no_grad():
    for images, targets in test_loader:
      images, targets = images.cuda(), targets.cuda()
      # adversarial
      if adv:
        images = adv(net, images, targets)
      logits = net(images)
      loss = F.cross_entropy(logits, targets)
      pred = logits.data.max(1)[1]
      total_loss += float(loss.data)
      total_correct += pred.eq(targets.data).sum().item()

  return total_loss / len(test_loader), total_correct / len(
      test_loader.dataset)


def test_c(net, test_data, base_path):
  """Evaluate network on given corrupted dataset."""
  corruption_accs = []
  corrs = CBAR_CORRUPTIONS if 'Bar' in base_path else CORRUPTIONS
  for corruption in corrs:
    # Reference to original data is mutated
    test_data.data = np.load(base_path + corruption + '.npy')
    test_data.targets = torch.LongTensor(np.load(base_path + 'labels.npy'))

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True)

    test_loss, test_acc = test(net, test_loader)
    corruption_accs.append(test_acc)
    print('{}\tTest Loss {:.3f} | Test Error {:.3f}'.format(
        corruption, test_loss, 100 - 100. * test_acc))

  return np.mean(corruption_accs)

def normalize_l2(x):
  """
  Expects x.shape == [N, C, H, W]
  """
  norm = torch.norm(x.view(x.size(0), -1), p=2, dim=1)
  norm = norm.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
  return x / norm

class PGD(nn.Module):
    def __init__(self, epsilon, num_steps, step_size, grad_sign=True):
        super().__init__()
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.grad_sign = grad_sign

    def forward(self, model, bx, by):
        """
        :param model: the classifier's forward method
        :param bx: batch of images
        :param by: true labels
        :return: perturbed batch of images
        """
        # unnormalize
        bx = (bx+1)/2

        adv_bx = bx.detach()
        adv_bx += torch.zeros_like(adv_bx).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            adv_bx.requires_grad_()
            with torch.enable_grad():
                logits = model(adv_bx * 2 - 1)
                loss = F.cross_entropy(logits, by, reduction='sum')
            grad = torch.autograd.grad(loss, adv_bx, only_inputs=True)[0]

            if self.grad_sign:
                adv_bx = adv_bx.detach() + self.step_size * torch.sign(grad.detach())
            else:
                grad = normalize_l2(grad.detach())
                adv_bx = adv_bx.detach() + self.step_size * grad

            adv_bx = torch.min(torch.max(adv_bx, bx - self.epsilon), bx + self.epsilon).clamp(0, 1)

        return adv_bx*2-1

def main():
  torch.manual_seed(1)
  np.random.seed(1)

  # Load datasets
  train_transform = transforms.Compose(
      [transforms.RandomHorizontalFlip(),
       transforms.RandomCrop(32, padding=4)])
  mixing_set_transform = transforms.Compose(
      [transforms.Resize(36), 
       transforms.RandomCrop(32)])
  to_tensor = transforms.ToTensor()
  normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)
  test_transform = transforms.Compose(
      [transforms.ToTensor(), normalize])

  if args.dataset == 'cifar10':
    train_data = datasets.CIFAR10(
        os.path.join(args.data_path, 'cifar'), train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR10(
        os.path.join(args.data_path, 'cifar'), train=False, transform=test_transform, download=True)
    base_c_path = os.path.join(args.data_path, 'cifar/CIFAR-10-C/')
    base_c_bar_path = os.path.join(args.data_path, 'cifar/CIFAR-10-C-Bar/')
    num_classes = 10
  else:
    train_data = datasets.CIFAR100(
        os.path.join(args.data_path, 'cifar'), train=True, transform=train_transform, download=True)
    test_data = datasets.CIFAR100(
        os.path.join(args.data_path, 'cifar'), train=False, transform=test_transform, download=True)
    base_c_path = os.path.join(args.data_path, 'cifar/CIFAR-100-C/')
    base_c_bar_path = os.path.join(args.data_path, 'cifar/CIFAR-100-C-Bar/')
    num_classes = 100

  if args.use_300k:
    mixing_set = RandomImages300K(file='300K_random_images.npy', transform=transforms.Compose(
      [transforms.ToTensor(), transforms.ToPILImage(), transforms.RandomCrop(32, padding=4),
      transforms.RandomHorizontalFlip()]))
  else:
    mixing_set = datasets.ImageFolder(args.mixing_set, transform=mixing_set_transform)
  print('train_size', len(train_data))
  print('aug_size', len(mixing_set))

  train_data = PixMixDataset(train_data, mixing_set, {'normalize': normalize, 'tensorize': to_tensor})

  # Fix dataloader worker issue
  # https://github.com/pytorch/pytorch/issues/5059
  def wif(id):
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))

  train_loader = torch.utils.data.DataLoader(
      train_data,
      batch_size=args.batch_size,
      shuffle=True,
      num_workers=args.num_workers,
      pin_memory=True,
      worker_init_fn=wif)

  test_loader = torch.utils.data.DataLoader(
      test_data,
      batch_size=args.eval_batch_size,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True)

  # Create model
  if args.model == 'densenet':
    net = densenet(num_classes=num_classes)
  elif args.model == 'wrn':
    net = WideResNet(args.layers, num_classes, args.widen_factor, args.droprate)
  elif args.model == 'resnext':
    net = resnext29(num_classes=num_classes)

  optimizer = torch.optim.SGD(
      net.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.decay,
      nesterov=True)

  # Distribute model across all visible GPUs
  net = torch.nn.DataParallel(net).cuda()
  cudnn.benchmark = True

  # initialize adversary
  adversary = PGD(epsilon=2./255, num_steps=20, step_size=0.5/255).cuda()

  start_epoch = 0

  if args.resume:
    if os.path.isfile(args.resume):
      checkpoint = torch.load(args.resume)
      start_epoch = checkpoint['epoch'] + 1
      best_acc = checkpoint['best_acc']
      net.load_state_dict(checkpoint['state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer'])
      print('Model restored from epoch:', start_epoch)

  if args.evaluate:
    # Evaluate clean accuracy first because test_c mutates underlying data
    test_loss, test_acc = test(net, test_loader)
    print('Clean\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
        test_loss, 100 - 100. * test_acc))

    adv_test_loss, adv_test_acc = test(net, test_loader, adv=adversary)
    print('Adversarial\n\tTest Loss {:.3f} | Test Error {:.2f}'.format(
        adv_test_loss, 100 - 100. * adv_test_acc))

    # test_c_acc = test_c(net, test_data, base_c_path)
    # print('Mean Corruption Error: {:.3f}'.format(100 - 100. * test_c_acc))
    return

  scheduler = torch.optim.lr_scheduler.LambdaLR(
      optimizer,
      lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
          step,
          args.epochs * len(train_loader),
          1,  # lr_lambda computes multiplicative factor
          1e-6 / args.learning_rate))

  if not os.path.exists(args.save):
    os.makedirs(args.save)
  elif args.save != 'snapshots':
    raise Exception('%s exists' % args.save)
  if not os.path.isdir(args.save):
    raise Exception('%s is not a dir' % args.save)

  log_path = os.path.join(args.save,
                          args.dataset + '_' + args.model + '_training_log.csv')
  with open(log_path, 'w') as f:
    f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

  best_acc = 0
  print('Beginning training from epoch:', start_epoch + 1)
  for epoch in range(start_epoch, args.epochs):
    begin_time = time.time()

    train_loss_ema = train(net, train_loader, optimizer, scheduler)
    test_loss, test_acc = test(net, test_loader)

    is_best = test_acc > best_acc
    best_acc = max(test_acc, best_acc)
    checkpoint = {
        'epoch': epoch,
        'dataset': args.dataset,
        'model': args.model,
        'state_dict': net.state_dict(),
        'best_acc': best_acc,
        'optimizer': optimizer.state_dict(),
    }

    save_path = os.path.join(args.save, 'checkpoint.pth.tar')
    torch.save(checkpoint, save_path)
    if is_best:
      shutil.copyfile(save_path, os.path.join(args.save, 'model_best.pth.tar'))

    with open(log_path, 'a') as f:
      f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' % (
          (epoch + 1),
          time.time() - begin_time,
          train_loss_ema,
          test_loss,
          100 - 100. * test_acc,
      ))

    print(
        'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} |'
        ' Test Error {4:.2f}'
        .format((epoch + 1), int(time.time() - begin_time), train_loss_ema,
                test_loss, 100 - 100. * test_acc))

  _, adv_test_acc = test(net, test_loader, adv=adversary)
  print('Adversarial Test Error: {:.3f}\n'.format(100 - 100. * adv_test_acc))
  
  test_c_acc = test_c(net, test_data, base_c_path)
  print('Mean C Corruption Error: {:.3f}\n'.format(100 - 100. * test_c_acc))

  # test_c_bar_acc = test_c(net, test_data, base_c_bar_path)
  # print('Mean C-Bar Corruption Error: {:.3f}\n'.format(100 - 100. * test_c_bar_acc))

  # print('Mean Corruption Error: {:.3f}\n'.format(100 - 100. * (15*test_c_acc + 10*test_c_bar_acc)/25))

  with open(log_path, 'a') as f:
    f.write('%03d,%05d,%0.6f,%0.5f,%0.2f\n' %
            (args.epochs + 1, 0, 0, 0, 100 - 100 * test_c_acc))


if __name__ == '__main__':
  main()
