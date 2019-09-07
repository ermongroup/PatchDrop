"""
This function jointly finetunes the policy network and high resolution classifier
using only high resolution classifier. Ideally, you should load the pre-trained model
as described in the paper.
How to Run on the CIFAR10 and CIFAR100 Datasets:
    python finetune.py --model R32_C10, R32_C100
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 256
       --Load Load from the latest checkpoint (agent)
       --ckpt_hr_cl Load from the latest checkpoint (hr_classifier)
How to Run on the fMoW Dataset:
    python finetune.py --model R34_fMoW
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 256
       --Load Load from the latest checkpoint (agent)
       --ckpt_lr_cl Load from the latest checkpoint (lr_classifier)
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
from torch.distributions import Bernoulli

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='PatchDrop Finetuning-I')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--ckpt_hr_cl', help='checkpoint directory for the high resolution classifier')
parser.add_argument('--load', default=None, help='checkpoint to load pretrained agent')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=1600, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=2000, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--penalty', type=float, default=-5, help='penalty for classifier mistakes')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor-helps with exploration')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):
    # This steps trains the policy network and high resolution classifier jointly
    # rnet represents the high resolution classifier
    # rnet_lr represents the low resolution classifier
    agent.train()
    rnet.eval()

    matches, rewards, rewards_baseline, policies = [], [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()
        inputs_agent = inputs.clone()

        # Get the low resolution agent images
        inputs_agent = utils.agent_input(inputs_agent, interval, img_size)
        probs, value = agent(inputs_agent)
        probs = probs*args.alpha + (1-probs)*(1-args.alpha)

        # Sample the policies from the Bernoulli distribution characterized by agent's output
        distr = Bernoulli(probs)
        policy_sample = distr.sample()

        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        inputs_map = inputs.clone()
        inputs_sample = inputs.clone()

        # Agent sampled high resolution images
        inputs_map = utils.agent_chosen_input(inputs_map, policy_map, mappings, interval)
        inputs_sample = utils.agent_chosen_input(inputs_sample, policy_sample.int(), mappings, interval)
        v_inputs_map = Variable(inputs_map.data, volatile=True)
        v_inputs = Variable(inputs_sample.data, volatile=True)

        # Get the predictions for baseline and sampled policy
        preds_map = rnet.forward(v_inputs_map)
        preds_sample = rnet.forward(v_inputs)

        # Get the rewards for both policies
        reward_map, match = utils.compute_reward(preds_map, targets, policy_map.data, args.penalty)
        reward_sample, _ = utils.compute_reward(preds_sample, targets, policy_sample.data, args.penalty)

        # Find the joint loss from the classifier and agent
        advantage = reward_sample - reward_map
        loss = -distr.log_prob(policy_sample).sum(1, keepdim=True) * Variable(advantage)
        loss = loss.sum()
        loss += F.cross_entropy(preds_sample, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        matches.append(match.cpu())
        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_map.cpu())
        policies.append(policy_sample.data.cpu())

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)
    print 'Train: %d | Acc: %.3f | Rw: %.2E | S: %.3f | V: %.3f | #: %d'%(epoch, accuracy, reward, sparsity, variance, len(policy_set))

    log_value('train_accuracy', accuracy, epoch)
    log_value('train_reward', reward, epoch)
    log_value('train_sparsity', sparsity, epoch)
    log_value('train_variance', variance, epoch)
    log_value('train_baseline_reward', torch.cat(rewards_baseline, 0).mean(), epoch)
    log_value('train_unique_policies', len(policy_set), epoch)

def test(epoch):
    # Test the policy network and the high resolution classifier
    agent.eval()
    rnet.eval()

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs, targets = Variable(inputs, volatile=True), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()
        inputs_agent = inputs.clone()

        # Get the low resolution agent images
        inputs_agent = utils.agent_input(inputs_agent, interval, img_size)
        probs, value = agent(inputs_agent)

        # Sample Test time Policy Using Bernoulli Distribution
        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0
        policy = Variable(policy)

        # Get the Agent Determined Images
        inputs = utils.agent_chosen_input(inputs, policy, mappings, interval)
        v_inputs = Variable(inputs.data, volatile=True)

        # Get the predictions from the high resolution classifier
        preds = rnet.forward(v_inputs)

        # Get the reward for the sampled policy and given predictions
        reward, match = utils.compute_reward(preds, targets, policy.data, args.penalty)

        matches.append(match)
        rewards.append(reward)
        policies.append(policy.data)

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    print 'Test - Acc: %.3f | Rw: %.2E | S: %.3f | V: %.3f | #: %d'%(accuracy, reward, sparsity, variance, len(policy_set))

    log_value('test_accuracy', accuracy, epoch)
    log_value('test_reward', reward, epoch)
    log_value('test_sparsity', sparsity, epoch)
    log_value('test_variance', variance, epoch)
    log_value('test_unique_policies', len(policy_set), epoch)

    # save the model --- policy network and high resolution classifier
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    rnet_state_dict = rnet.module.state_dict() if args.parallel else rnet.state_dict()

    state = {
      'agent': agent_state_dict,
      'resnet_hr': rnet_state_dict,
      'epoch': epoch,
      'reward': reward,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f_R_%.2E'%(epoch, accuracy, reward))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
rnet, _, agent = utils.get_model(args.model)
rnet.cuda()
agent.cuda()

# ------- PatchDrop Action Space for fMoW -----------------------
num_patches = 16 # Fixed in the paper, but can be changed
mappings, img_size, interval = utils.action_space_model(args.model.split('_')[1])

if args.load is not None:
    checkpoint = torch.load(args.load)
    key = 'net' if 'net' in checkpoint else 'agent'
    agent.load_state_dict(checkpoint[key])
    print 'loaded pretrained model from', args.load

if args.ckpt_hr_cl is not None:
    checkpoint = torch.load(args.ckpt_hr_cl)
    rnet.load_state_dict(checkpoint['state_dict'])
    print 'loaded the high resolution classifier'

if args.parallel:
    agent = nn.DataParallel(agent)
    rnet = nn.DataParallel(rnet)

# Update the parameters of the policy network and high resolution classifier
optimizer = optim.Adam(list(agent.parameters())+list(rnet.parameters()), lr=args.lr)

configure(args.cv_dir+'/log', flush_secs=5)
start_epoch = 0
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch%5==0:
        test(epoch)
