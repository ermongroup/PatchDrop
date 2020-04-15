"""
This function pretrains the policy network using the high resolution classifier
output-explained as pretraining the policy network in the paper.
How to Run on Different Benchmarks:
    python pretrain.py --model R32_C10, R32_C100, R34_fMoW, R50_ImgNet
       --lr 1e-4
       --cv_dir checkpoint directory
       --batch_size 1048 (Higher is better)
       --ckpt_hr_cl Load the checkpoint from the directory for HR classifier
       --lr_size 8, 56 (Depends on the dataset)
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
import torch.optim as optim

from torch.distributions import Multinomial, Bernoulli
from utils import utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='Policy Network Finetuning-I')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--ckpt_hr_cl', help='checkpoint directory for the high resolution classifier')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--penalty', type=float, default=-0.5, help='to penalize the PN for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--lr_size', type=int, default=8, help='Policy Network Image Size')
parser.add_argument('--test_interval', type=int, default=5, help='At what epoch to test the model')
args = parser.parse_args()

if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(__file__, args)

def train(epoch):
    # This steps trains the policy network only
    agent.train()

    matches, rewards, rewards_baseline, policies = [], [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        inputs, targets = Variable(inputs), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()
        inputs_agent = inputs.clone()
        inputs_map = inputs.clone()
        inputs_sample = inputs.clone()

        # Run the low-res image through Policy Network
        inputs_agent = torch.nn.functional.interpolate(inputs_agent, (args.lr_size, args.lr_size))
        probs = F.sigmoid(agent.forward(inputs_agent, args.model.split('_')[1], 'lr'))
        probs = probs*args.alpha + (1-args.alpha) * (1-probs)

        # Sample the policies from the Bernoulli distribution characterized by agent's output
        distr = Bernoulli(probs)
        policy_sample = distr.sample()

        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0

        # Agent sampled high resolution images
        inputs_map = utils.agent_chosen_input(inputs_map, policy_map, mappings, patch_size)
        inputs_sample = utils.agent_chosen_input(inputs_sample, policy_sample.int(), mappings, patch_size)

        # Forward propagate images through the classifiers
        preds_map = rnet.forward(inputs_map, args.model.split('_')[1], 'hr')
        preds_sample = rnet.forward(inputs_sample, args.model.split('_')[1], 'hr')

        # Find the reward for baseline and sampled policy
        reward_map, match = utils.compute_reward(preds_map, targets, policy_map.data, args.penalty)
        reward_sample, _ = utils.compute_reward(preds_sample, targets, policy_sample.data, args.penalty)
        advantage = reward_sample.cuda().float() - reward_map.cuda().float()

        # Find the loss for only the policy network
        loss = -distr.log_prob(policy_sample)
        loss = loss * Variable(advantage).expand_as(policy_sample)
        loss = loss.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        matches.append(match.cpu())
        rewards.append(reward_sample.cpu())
        rewards_baseline.append(reward_map.cpu())
        policies.append(policy_sample.data.cpu())

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    print('Train: %d | Acc: %.3f | Rw: %.2E | S: %.3f | V: %.3f | #: %d'%(epoch, accuracy, reward, sparsity, variance, len(policy_set)))
    log_value('train_accuracy', accuracy, epoch)
    log_value('train_reward', reward, epoch)
    log_value('train_sparsity', sparsity, epoch)
    log_value('train_variance', variance, epoch)
    log_value('train_baseline_reward', torch.cat(rewards_baseline, 0).mean(), epoch)
    log_value('train_unique_policies', len(policy_set), epoch)


def test(epoch):

    agent.eval()

    matches, rewards, policies = [], [], []
    for batch_idx, (inputs, targets) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        inputs, targets = Variable(inputs, volatile=True), Variable(targets).cuda(async=True)
        if not args.parallel:
            inputs = inputs.cuda()

        # Get the low resolution agent images
        inputs_agent = inputs.clone()
        inputs_agent = torch.nn.functional.interpolate(inputs_agent, (args.lr_size, args.lr_size))
        probs = F.sigmoid(agent.forward(inputs_agent, args.model.split('_')[1], 'lr'))

        # Sample the test-time policy
        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0

        # Get the masked high-res image and perform inference
        inputs = utils.agent_chosen_input(inputs, policy, mappings, patch_size)
        preds = rnet.forward(inputs, args.model.split('_')[1], 'hr')

        reward, match = utils.compute_reward(preds, targets, policy.data, args.penalty)

        matches.append(match)
        rewards.append(reward)
        policies.append(policy.data)

    accuracy, reward, sparsity, variance, policy_set = utils.performance_stats(policies, rewards, matches)

    print('Test - Acc: %.3f | Rw: %.2E | S: %.3f | V: %.3f | #: %d'%(accuracy, reward, sparsity, variance, len(policy_set)))
    log_value('test_accuracy', accuracy, epoch)
    log_value('test_reward', reward, epoch)
    log_value('test_sparsity', sparsity, epoch)
    log_value('test_variance', variance, epoch)
    log_value('test_unique_policies', len(policy_set), epoch)

    # Save the Policy Network - Classifier is fixed in this phase
    agent_state_dict = agent.module.state_dict() if args.parallel else agent.state_dict()
    state = {
      'agent': agent_state_dict,
      'epoch': epoch,
      'reward': reward,
      'acc': accuracy
    }
    torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f_R_%.2E'%(epoch, accuracy, reward))

#--------------------------------------------------------------------------------------------------------#
trainset, testset = utils.get_dataset(args.model, args.data_dir)
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=8)
rnet, _, agent = utils.get_model(args.model)

# Save the args to the checkpoint directory
configure(args.cv_dir+'/log', flush_secs=5)

# Agent Action Space
mappings, _, patch_size = utils.action_space_model(args.model.split('_')[1])

# Load the classifier - has to exist
checkpoint = torch.load(args.ckpt_hr_cl)
rnet.load_state_dict(checkpoint['state_dict'])
print('loaded the high resolution classifier')

# Load the Policy Network
start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print('loaded agent from', args.load)

# Parallelize the models if multiple GPUs available - Important for Large Batch Size
if args.parallel:
    agent = nn.DataParallel(agent)
    rnet = nn.DataParallel(rnet)

rnet.eval().cuda() # HR Classifier is Fixed
agent.cuda() # Only agent is updated

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# Start training and testing
for epoch in range(start_epoch, start_epoch+args.max_epochs):
    train(epoch)
    if epoch % args.test_interval == 0:
        test(epoch)

