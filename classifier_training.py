"""
This function pretrains the high and low resolution classifiers.
How to Run on the CIFAR10 and CIFAR100 Datasets:
    python classifer_training.py --model R32_C10, R32_C100
       --lr 1e-1
       --cv_dir checkpoint directory
       --batch_size 64
       --wd 5e-4
How to Run on the fMoW Dataset(Uses ImageNet pretrained model):
    python classifier_training.py --model R34_fMoW
       --lr 1e-1
       --cv_dir checkpoint directory
       --batch_size 64
       --wd 5e-4
"""
import os
from tensorboard_logger import configure, log_value
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.utils.data as torchdata
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm
import utils
import torch.optim as optim
import pdb
from torch.distributions import Multinomial, Bernoulli
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='PatchSelector Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--beta', type=float, default=1e-1, help='entropy multiplier')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--stream_id', default='HR', help='high resolution or loW resolution classifier')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):

    rnet.train()
    matches, losses = [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        if not args.parallel:
    	    inputs = inputs.cuda()

        inputs = utils.agent_input(inputs, img_size, inputs.size(2))
        v_inputs = Variable(inputs.data, volatile=True)

        preds = rnet.forward(v_inputs)

        _, pred_idx = preds.max(1)
        match = (pred_idx==targets).data

        loss = F.cross_entropy(preds, targets)
        # --------------------------------------------------------------------------

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        matches.append(match.cpu())
        losses.append(loss.cpu())

    accuracy = torch.cat(matches, 0).float().mean()
    loss = torch.stack(losses).mean()
    log_str = 'E: %d | A: %.3f | L: %.3f'%(epoch, accuracy, loss)
    print log_str

    log_value('train_accuracy', accuracy, epoch)
    log_value('train_loss', loss, epoch)

def test(epoch):

    rnet.eval()
    matches = []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs, targets = Variable(inputs, volatile=True), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        inputs = utils.agent_input(inputs, img_size, inputs.size(2))
        v_inputs = Variable(inputs.data, volatile=True)

        preds = rnet.forward(v_inputs)

        _, pred_idx = preds.max(1)
        match = (pred_idx==targets).data

        matches.append(match.cpu())

    accuracy = torch.cat(matches, 0).float().mean()
    log_str = 'TS: %d | A: %.3f'%(epoch, accuracy)
    print log_str

    log_value('train_accuracy', accuracy, epoch)

    # save the model parameters
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()

    state = {
      'state_dict': rnet_state_dict,
      'epoch': epoch,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f'%(epoch, accuracy))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
rnet, _, _ = utils.get_model(args.model)
rnet.cuda()

# Get the image size to train the CNN on - low resolution or high resolution
start_epoch = 0
if args.stream_id == 'HR':
    if args.model.split('_')[1] == 'fMoW':
        img_size = 112
    else:
        img_size = 32
else:
    if args.model.split('_')[1] == 'fMoW':
        img_size = 28
    else:
        img_size = 8

optimizer = optim.SGD(rnet.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [150, 250])
configure(args.cv_dir+'/log', flush_secs=5)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch % 10 == 0:
        test(epoch)
    lr_scheduler.step()
