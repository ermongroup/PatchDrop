"""
This function pretrains the policy network using the high resolution classifier
output-explained as pretraining the policy network in the paper.
How to Run on the CIFAR10 and CIFAR100 Datasets:
    python pretrain.py --model R32_C10, R32_C100
       --lr 1e-3
       --cv_dir checkpoint directory
       --batch_size 512
       --ckpt_hr_cl Load the checkpoint from the directory (hr_classifier)
How to Run on the fMoW Dataset:
    python pretrain.py --model R34_fMoW
       --lr 1e-3
       --cv_dir checkpoint directory
       --batch_size 1024
       --ckpt_hr_cl Load the checkpoint from the directory (hr_classifier)
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
from torch.distributions import Multinomial, Bernoulli
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser(description='PatchDrop Pre-Training')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--model', default='R32_C10', help='R<depth>_<dataset> see utils.py for a list of configurations')
parser.add_argument('--ckpt_hr_cl', help='checkpoint directory for the high resolution classifier')
parser.add_argument('--data_dir', default='data/', help='data directory')
parser.add_argument('--load', default=None, help='checkpoint to load agent from')
parser.add_argument('--cv_dir', default='cv/tmp/', help='checkpoint directory (models and logs are saved here)')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--epoch_step', type=int, default=10000, help='epochs after which lr is decayed')
parser.add_argument('--max_epochs', type=int, default=10000, help='total epochs to run')
parser.add_argument('--parallel', action ='store_true', default=False, help='use multiple GPUs for training')
parser.add_argument('--penalty', type=float, default=-0.5, help='gamma: reward for incorrect predictions')
parser.add_argument('--alpha', type=float, default=0.8, help='probability bounding factor')
parser.add_argument('--sigma', type=float, default=0.1, help='multiplier for the entropy loss')
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
        inputs_agent = inputs.clone()
        inputs_map = inputs.clone()
        inputs_sample = inputs.clone()
        if not args.parallel:
    	    inputs = inputs.cuda()
            inputs_agent = inputs_agent.cuda()
            inputs_map = inputs_map.cuda()
            inputs_sample = inputs_sample.cuda()

        # Get the low resolution agent images
        inputs_agent = utils.agent_input(inputs_agent, interval, img_size)
        probs, value = agent(inputs_agent)
        probs = probs*args.alpha + (1-args.alpha) * (1-probs)

        # Sample the policies from the Bernoulli distribution characterized by agent's output
        distr = Bernoulli(probs)
        policy_sample = distr.sample()

        # Test time policy - used as baseline policy in the training step
        policy_map = probs.data.clone()
        policy_map[policy_map<0.5] = 0.0
        policy_map[policy_map>=0.5] = 1.0
        policy_map = Variable(policy_map)

        # Use curriculum learning, activate one patch for 0<K<16
        if epoch< num_patches:
            policy_sample[:, epoch+1:] = 1
            policy_map[:, epoch+1:] = 1
            policy_mask = Variable(torch.ones(inputs.size(0), policy_sample.size(1))).cuda()
            policy_mask[:, epoch+1:] = 0
        else:
            policy_mask = None

        # Agent sampled high resolution images
        inputs_map = utils.agent_chosen_input(inputs_map, policy_map, mappings, interval)
        inputs_sample = utils.agent_chosen_input(inputs_sample, policy_sample.int(), mappings, interval)
        v_inputs_map = Variable(inputs_map.data, volatile=True)
        v_inputs = Variable(inputs_sample.data, volatile=True)

        # Forward propagate images through the classifiers
        preds_map = rnet.forward(v_inputs_map)
        preds_sample = rnet.forward(v_inputs)

        # Find the reward for baseline and sampled policy
        reward_map, match = utils.compute_reward(preds_map, targets, policy_map.data, args.penalty)
        reward_sample, _ = utils.compute_reward(preds_sample, targets, policy_sample.data, args.penalty)
        advantage = reward_sample.cuda().float() - reward_map.cuda().float()

        # Find the loss for only the policy network
        loss = -distr.log_prob(policy_sample)
        loss = loss * Variable(advantage).expand_as(policy_sample)

        # mask for curriculum learning
        if policy_mask is not None:
            loss = policy_mask * loss
        loss = loss.mean()

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
        inputs_agent = inputs.clone()
        if not args.parallel:
            inputs = inputs.cuda()
            inputs_agent = inputs_agent.cuda()

        # Get the low resolution agent images
        inputs_agent = utils.agent_input(inputs_agent, interval, img_size)
        probs, value = agent(inputs_agent)

        # Sample the policy from the agents output
        policy = probs.data.clone()
        policy[policy<0.5] = 0.0
        policy[policy>=0.5] = 1.0
        policy = Variable(policy)

        # Apply curriculum learning mask
        if epoch < num_patches:
            policy[:, epoch+1:] = 1

        # Get the agent sampled high resolution image and perform inference
        inputs = utils.agent_chosen_input(inputs, policy, mappings, interval)
        preds = rnet.forward(inputs)

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

    # save the model --- agent
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
trainloader = torchdata.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=16)
testloader = torchdata.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
rnet, _, agent = utils.get_model(args.model)

# ------- PatchDrop Action Space for fMoW -----------------------
num_patches = 16 # Fixed in the paper, but can be changed
mappings, img_size, interval = utils.action_space_model(args.model.split('_')[1])

start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    agent.load_state_dict(checkpoint['agent'])
    start_epoch = checkpoint['epoch'] + 1
    print 'loaded agent from', args.load

if args.ckpt_hr_cl is not None:
    checkpoint = torch.load(args.ckpt_hr_cl)
    # rnet.load_state_dict(checkpoint['state_dict'])
    utils.load_weights_to_flatresnet(checkpoint, rnet)
    print 'loaded the high resolution classifier'

# Parallelize the models if multiple GPUs available - Important for Large Batch Size
if args.parallel:
    agent = nn.DataParallel(agent)
    rnet = nn.DataParallel(rnet)

rnet.cuda()
agent.cuda()

# Update the parameters of the policy network
optimizer = optim.Adam(agent.parameters(), lr=args.lr)

# Save the args to the checkpoint directory
configure(args.cv_dir+'/log', flush_secs=5)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    if epoch < num_patches:
        print 'curriculum learning - training on the %d patches ...' % epoch
    elif epoch==num_patches:
        print 'curriculum learning is over'
    else:
        print 'all the patches activated'

    train(epoch)
    if epoch % 10 == 0:
        test(epoch)
