'''
******  Weight Direction Experiment  ******
parallel with direction.0.0.3.py
calculate angular update every iteration
'''
from __future__ import print_function
# from __future__ import division

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import os
import argparse
import logging
import time
import shutil
import random
import sys
import numpy as np
# np.set_printoptions(threshold=np.inf)
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
# import tensorboard_logger
from tensorboard_logger.tensorboard_logger import configure, unconfigure, log_value
# from torch.utils.tensorboard import SummaryWriter
import models
import data_loader
# import active_util
import pdb
import utils
import torchsummary

# Training
def train(logger, train_loader, model, criterion, optimizer, epoch, update_freq, lr_schedule, conv_parameters, 
            prev_weight_iter, prev_conv_weight_iter, prev_l2_norm_iter, prev_conv_l2_norm_iter):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    iters_per_epoch = len(train_loader) // update_freq
    # epoch 마다 angular update는 그대로 계산해야 되니까 iteration용은 따로 _iter 붙은 변수로 설정 -> 어차피 train function 밖에서 계산해서 필요 없음

    wt_theta_iter_one_epoch, wt_conv_theta_iter_one_epoch = [],[] # for one epoch average

    # model.train()
    if args.amp:
        scaler = GradScaler()
    # optimizer.zero_grad()
    for param in model.parameters():
        param.grad = None

    end = time.time()        
    for i, (inputs, targets) in enumerate(train_loader):
        model.train()
        inputs, targets = inputs, targets.cuda()

        optim_iter = i // update_freq
        it = iters_per_epoch * epoch + optim_iter  # global training iteration
        for k, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr_schedule[it]
            # momentum 출력!
            # if i > 0:
            #     for p in param_group['params']:
            #         if p.requires_grad:
            #             print((optimizer.state[p])['momentum_buffer'].size())

        # compute output
        if args.amp:
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets) 

        # measure accuracy and record loss
        acc_temp, loss_temp = 0, 0
        acc1 = accuracy(outputs.data, targets, topk=(1,))[0]
        if update_freq == 1 :
            losses.update(loss.data, inputs.size(0))    
            top1.update(acc1, inputs.size(0))
        elif (i+1) % update_freq != 0:
            acc_temp += accuracy(outputs.data, targets, topk=(1,))[0]
            loss_temp += loss.data
        else:
            acc1 = accuracy(outputs.data, targets, topk=(1,))[0]
            losses.update((loss.data+loss_temp)/update_freq, inputs.size(0)*update_freq)
            top1.update((acc1+acc_temp)/update_freq, inputs.size(0)*update_freq)

        # compute gradient and do SGD step
        if args.amp:    # for backward compatibility 
            scaler.scale(loss / update_freq).backward()
            if (i+1) % update_freq == 0 :
                if args.max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                if i+1 != len(train_loader):
                    # optimizer.zero_grad()
                    for param in model.parameters():
                        param.grad = None
        else:
            (loss / update_freq).backward()
            if (i+1) % update_freq == 0 :
                if args.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)    
                optimizer.step()
                if i+1 != len(train_loader):
                    # optimizer.zero_grad()
                    for param in model.parameters():
                        param.grad = None
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'lr {lr:.4f}'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            loss=losses, top1=top1, lr=lr_schedule[it]))

        if (it+1) % args.angle_freq == 0:
            model.eval()      
            wt_l2_norm, conv_l2_norm = 0,0
            wt_weight, wt_conv_weight = [],[]
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if not 'linear' in name and not 'classifier' in name:   # except for fully connected layer
                        norm_square = param.data.norm(2).item()**2
                        weight_vector = param.view(-1).data
                        # all weight 
                        wt_l2_norm = wt_l2_norm + norm_square
                        wt_weight.append(weight_vector)
                        # conv weight
                        if any(conv_parameter in name for conv_parameter in conv_parameters):
                            conv_l2_norm = conv_l2_norm + norm_square
                            wt_conv_weight.append(weight_vector)
                wt_weight = torch.cat(wt_weight,0)
                wt_conv_weight = torch.cat(wt_conv_weight,0)
                wt_l2_norm = wt_l2_norm**(1./2)
                conv_l2_norm = conv_l2_norm**(1./2)

                # calculate angular update in the degree
                wt_cos_theta_iter = torch.clamp(torch.dot(prev_weight_iter, wt_weight)/(prev_l2_norm_iter*wt_l2_norm),-1,1) # clamp for stability
                wt_theta_iter = (torch.acos(wt_cos_theta_iter)*(360./(2.*np.pi))).item()
                
                wt_conv_cos_theta_iter = torch.clamp(torch.dot(prev_conv_weight_iter, wt_conv_weight)/(prev_conv_l2_norm_iter*conv_l2_norm),-1,1) # clamp for stability
                wt_conv_theta_iter = (torch.acos(wt_conv_cos_theta_iter)*(360./(2.*np.pi))).item()
                
                wt_theta_iter_one_epoch.append(wt_theta_iter)            # for one epoch average
                wt_conv_theta_iter_one_epoch.append(wt_conv_theta_iter)  # for one epoch average

                if args.tensorboard:
                    log_value('wt_theta_iter', wt_theta_iter, it//args.angle_freq)
                    log_value('wt_conv_theta_iter', wt_conv_theta_iter, it//args.angle_freq)

                # angle_freq 마다 weight vector랑 norm 구함
                prev_l2_norm_iter = wt_l2_norm
                prev_conv_l2_norm_iter = conv_l2_norm
                prev_weight_iter = wt_weight
                prev_conv_weight_iter = wt_conv_weight
                # 위에 4개를 return 한다음 다음 epoch에 넘겨주면 됨, save_theta_iter도 

    # 한 epoch 마다 계산
    # log to TensorBoard
    model.eval()      
    wt_l2_norm, grad_l2_norm, conv_l2_norm, conv_grad_l2_norm = 0,0,0,0
    wt_weight, wt_conv_weight = [],[]
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not 'linear' in name and not 'classifier' in name:   # except for fully connected layer
                norm_square = param.data.norm(2).item()**2
                grad_norm_square = param.grad.data.norm(2).item()**2
                weight_vector = param.view(-1).data
                # all weight 
                wt_l2_norm = wt_l2_norm + norm_square
                grad_l2_norm = grad_l2_norm + grad_norm_square
                wt_weight.append(weight_vector)
                # conv weight
                if any(conv_parameter in name for conv_parameter in conv_parameters):
                    conv_l2_norm = conv_l2_norm + norm_square
                    conv_grad_l2_norm = conv_grad_l2_norm + grad_norm_square
                    wt_conv_weight.append(weight_vector)
        wt_weight = torch.cat(wt_weight,0)
        wt_conv_weight = torch.cat(wt_conv_weight,0)
        effecitve_lr = lr_schedule[(epoch+1)*iters_per_epoch-1]/wt_l2_norm              # LR / norm^2
        effecitve_conv_lr = lr_schedule[(epoch+1)*iters_per_epoch-1]/conv_l2_norm       # LR / conv_norm^2
        wt_l2_norm = wt_l2_norm**(1./2)
        grad_l2_norm = grad_l2_norm**(1./2)
        conv_l2_norm = conv_l2_norm**(1./2)
        conv_grad_l2_norm = conv_grad_l2_norm**(1./2)

    logger.info('wt_l2_norm: {}\t grad_l2_norm: {}\t'.format(wt_l2_norm, grad_l2_norm))
    logger.info('conv_l2_norm: {}\t conv_grad_l2_norm: {}\t'.format(conv_l2_norm, conv_grad_l2_norm))

    log_value('train_loss', losses.avg, epoch)
    log_value('train_acc', top1.avg, epoch)
    log_value('learning_rate', lr_schedule[(epoch+1)*iters_per_epoch-1], epoch)
    log_value('weight_decay', args.weight_decay, epoch)
    # log_value('momentum_buffer', momentum_buffer, epoch)
    log_value('wt_l2_norm', wt_l2_norm, epoch+1)
    log_value('grad_l2_norm', grad_l2_norm, epoch)
    log_value('conv_l2_norm', conv_l2_norm, epoch+1)
    log_value('conv_grad_l2_norm', conv_grad_l2_norm, epoch)
    log_value('effecitve_lr', effecitve_lr, epoch)
    log_value('effecitve__conv_lr', effecitve_conv_lr, epoch) # 중간에 오타 있는데 기존에 있는거랑 tensorboard에서 같이 보려고 안고침
    log_value('wt_theta_iter_avg', torch.mean(torch.as_tensor(wt_theta_iter_one_epoch)), epoch)
    log_value('wt_conv_theta_iter_avg', torch.mean(torch.as_tensor(wt_conv_theta_iter_one_epoch)), epoch)

    return [wt_weight, wt_conv_weight, wt_l2_norm, conv_l2_norm, 
            prev_weight_iter, prev_conv_weight_iter, prev_l2_norm_iter, prev_conv_l2_norm_iter, 
            wt_theta_iter_one_epoch, wt_conv_theta_iter_one_epoch,
            effecitve_lr, effecitve_conv_lr, (100.-top1.avg).item()]

def test(logger, test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    # confusion_matrix = torch.zeros(args.num_classes, args.num_classes)
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs, targets.cuda()
            
            # compute output
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1 = accuracy(outputs.data, targets, topk=(1,))[0]
            losses.update(loss.data, inputs.size(0))
            top1.update(acc1.cpu(), inputs.size(0))
            
            # # compute class separate accuracy
            # for t, p in zip(targets.view(-1), preds.view(-1)):
            #     confusion_matrix[t.long(), p.long()] += 1

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('[{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                                i, len(test_loader), batch_time=batch_time, loss=losses,
                                top1=top1))

        # accuracy_each_class = confusion_matrix.diag()/confusion_matrix.sum(1)
        logger.info(' * acc@1 {top1.avg:.3f}'.format(top1=top1))
        # logger.info(' * accuracy of each classes: \n  {}'.format(accuracy_each_class))

    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)

    return (top1.avg).item()


def save_checkpoint(state, save_dir, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    filename = '{}/checkpoint.pth.tar'.format(save_dir)
    # filename = os.path.join(args.save_dir, filename)
    torch.save(state, filename)

    if is_best:
        # fn_best = os.path.join(args.save_dir, 'model_best.pth.tar')
        fn_best = '{}/model_best.pth.tar'.format(save_dir)
        shutil.copyfile(filename, fn_best)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""

    if epoch in args.decay_epoch:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay
            lr = param_group['lr']
            args.lr = lr
    elif epoch==0:
        lr = 0            # why???? maybe error but negligible
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
        log_value('weight_decay', args.weight_decay, epoch)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    # give weight decay to BN or bias
    # if 'resnet' in args.net_type:
    #     bn_parameters = ['bn', 'shortcut.1']
    # elif 'densenet' in args.net_type:
    #     bn_parameters = ['bn']

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        # print('{}\t shape: {}'.format(name, param.shape))
        if not param.requires_grad:
            continue

        if args.filter_bn_bias:
            # do not give weight decay to BN and bias
            if len(param.shape) == 1 or name in skip_list:
                no_decay.append(param)
                print(name)
            else:
                decay.append(param)
        else:
             # give weight decay to all param except for linear
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def main(args, seed):
    # testing
    save_dir = args.save_dir+'/'+str(seed)
    try:
        os.makedirs(save_dir)
        os.chmod(save_dir, 0o777)
    except OSError:
        pass

    if args.tensorboard: configure(save_dir)

    logger = logging.getLogger("js_logger")
    fileHandler = logging.FileHandler(save_dir+'/train.log')
    streamHandler = logging.StreamHandler()
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    logger.setLevel(logging.INFO)
    logger.info(args)

    len_dataset = 50000
    if args.dataset == 'cifar10' or args.dataset == 'svhn':
        args.num_classes = 10
    elif args.dataset == 'cifar100':
        args.num_classes = 100
    elif args.dataset == 'tinyimagenet':
        args.num_classes = 200
        len_dataset = 100000
    elif args.dataset == 'imagenet':
        args.num_classes = 1000
        len_dataset = 1281167
    elif args.dataset == 'stl10':
        args.num_classes = 10
        len_dataset = 5000

    update_freq = 1
    # if args.dataset == "imagenet" and args.batch_size >= 1024:
    #     update_freq = args.batch_size / 1024
    # elif 'resnet' in args.net_type and 'cifar' in args.dataset and args.batch_size > 16384:
    #     update_freq = args.batch_size / 16384
    # elif 'densenet' in args.net_type and 'cifar' in args.dataset and args.batch_size > 8192:
    #     update_freq = args.batch_size / 8192
    # else:
    #     update_freq = 1
    loader_batch_size = int(args.batch_size/update_freq)

    if args.load_dir:
        if os.path.isdir(args.load_dir):
            # Load checkpoint.
            print("=> loading checkpoint '{}'".format(args.load_dir))
            checkpoint = torch.load(args.load_dir+'/'+args.checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.load_dir))
    
    # Data
    print('==> Preparing data..')
    if args.num_sample > 0:
        labeled_idx = np.random.choice(len_dataset, args.num_sample, replace=False)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(labeled_idx)
        train_loader, test_loader = data_loader.getDataSet(args.dataset, loader_batch_size, args.dataroot, \
                sampler=sampler, drop_last=args.drop_last, num_workers=args.num_workers)
    else:
        train_loader, test_loader = data_loader.getDataSet(args.dataset, loader_batch_size, args.dataroot, \
                drop_last=args.drop_last, num_workers=args.num_workers)

    # Model
    print('==> Building model..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_to_call = getattr(models, args.net_type)
    model = model_to_call(num_classes=args.num_classes, zero_init_residual=args.zero_init_residual, amp=args.amp, 
                            eps=args.eps, effnet_default_init=True)
    model = model.to(device)
    # torchsummary.summary(model, (3, 256, 256),device='cuda')
    model = torch.nn.DataParallel(model)
    # print(model)
    # exit()

    if device == 'cuda' and args.cudnn:
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    lr = args.lr
    weight_decay = args.weight_decay

    # if weight_decay and args.filter_bn:
    parameters = add_weight_decay(model, weight_decay)
    # else:
    #     parameters = model.parameters()

    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    # optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=0.1)
    lr_schedule = utils.scheduler(args.lr, args.epochs,
        len(train_loader) // update_freq, warmup_epochs=args.warmup_epochs, start_warmup_value=1e-3, type = args.schedule_type,
        final_value = args.lr_end, decay_epoch=args.decay_epoch)

    # for save data
    best_acc = 0
    train_errors, val_errors = [],[]
    wt_norms, conv_norms = [],[]
    wt_thetas, wt_conv_thetas = [],[]
    w0_thetas, w0_conv_thetas = [],[]
    # 이거 이름 제대로 정하고 train 함수에서 ouput 하는 list 받아서 계속 concat 해야 됨
    effective_lrs, effective_conv_lrs = [],[]
    wt_theta_iters, wt_conv_theta_iters = [],[] # angular_freq 마다 theta 기록
    save_data = {'train_errors':train_errors, 'val_errors':val_errors, 
                'wt_norms':wt_norms, 'conv_norms':conv_norms, 
                'wt_thetas':wt_thetas, 'wt_conv_thetas':wt_conv_thetas, 
                'w0_thetas':w0_thetas, 'w0_conv_thetas':w0_conv_thetas,
                'wt_theta_iters':wt_theta_iters, 'wt_conv_theta_iters':wt_conv_theta_iters,
                'effective_lrs':effective_lrs, 'effective_conv_lrs':effective_conv_lrs}

    # for calculate angular update
    w0_l2_norm, w0_conv_l2_norm = 0,0
    w0_weight, w0_conv_weight = [],[]

    conv_parameters, bn_parameters = [], []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d)) and not ('fc' in name):
            conv_parameters.append(name)
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
            bn_parameters.append(name)

    # before train
    with torch.no_grad():
        for name, param in model.named_parameters():
            # make the network scale invariant
            if 'linear' in name or 'classifier' in name:
            # if 'linear' in name:
                if 'inv' in args.net_type:  # invariant
                    param.requires_grad = False
            else:
                if args.alpha_sqaure is not None:   # It should be None for ICML experiments 
                    param.mul_(np.sqrt(args.alpha_sqaure))
                norm_square = param.data.norm(2).item()**2
                weight_vector = param.view(-1).data
                w0_l2_norm = w0_l2_norm + norm_square    # without linear layer
                w0_weight.append(weight_vector)

                if any(conv_parameter in name for conv_parameter in conv_parameters):
                    # print(name)
                    w0_conv_l2_norm = w0_conv_l2_norm + norm_square
                    w0_conv_weight.append(weight_vector)
            
        w0_weight = torch.cat(w0_weight,0)
        w0_conv_weight = torch.cat(w0_conv_weight,0)
        w0_l2_norm = w0_l2_norm**(1./2)
        w0_conv_l2_norm = w0_conv_l2_norm**(1./2)
        prev_l2_norm = prev_l2_norm_iter = w0_l2_norm
        prev_conv_l2_norm = prev_conv_l2_norm_iter = w0_conv_l2_norm
        prev_weight = prev_weight_iter = w0_weight
        prev_conv_weight = prev_conv_weight_iter = w0_conv_weight
        wt_norms.append(w0_l2_norm)
        conv_norms.append(w0_conv_l2_norm)

    logger.info('wt_l2_norm {}\tconv_l2_norm {}'.format(w0_l2_norm, w0_conv_l2_norm))
    if args.tensorboard:
        log_value('wt_l2_norm', w0_l2_norm, 0)
        log_value('conv_l2_norm', w0_conv_l2_norm, 0)

    ######## epoch start ########
    for epoch in range(args.start_epoch, args.epochs):
        # scheduler.step(epoch)
        # adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        wt_weight, wt_conv_weight, wt_l2_norm, conv_l2_norm, \
        prev_weight_iter, prev_conv_weight_iter, prev_l2_norm_iter, prev_conv_l2_norm_iter, \
        wt_theta_iter_one_epoch, wt_conv_theta_iter_one_epoch, \
        effective_lr, effective_conv_lr, train_error \
                = train(logger, train_loader, model, criterion, optimizer, epoch, update_freq, lr_schedule, conv_parameters, \
                        prev_weight_iter, prev_conv_weight_iter, prev_l2_norm_iter, prev_conv_l2_norm_iter)
        
        # calculate angular update in the degree
        w0_theta = torch.acos(torch.dot(w0_weight, wt_weight)/(w0_l2_norm*wt_l2_norm)).item()*(360./(2.*np.pi))
        wt_cos_theta = torch.clamp(torch.dot(prev_weight, wt_weight)/(prev_l2_norm*wt_l2_norm),-1,1) # clamp for stability
        wt_theta = (torch.acos(wt_cos_theta)*(360./(2.*np.pi))).item()
        
        w0_conv_theta = torch.acos(torch.dot(w0_conv_weight, wt_conv_weight)/(w0_conv_l2_norm*conv_l2_norm)).item()*(360./(2.*np.pi))
        wt_conv_cos_theta = torch.clamp(torch.dot(prev_conv_weight, wt_conv_weight)/(prev_conv_l2_norm*conv_l2_norm),-1,1) # clamp for stability
        wt_conv_theta = (torch.acos(wt_conv_cos_theta)*(360./(2.*np.pi))).item()

        train_errors.append(train_error)
        wt_norms.append(wt_l2_norm)
        conv_norms.append(conv_l2_norm)
        effective_lrs.append(effective_lr)
        effective_conv_lrs.append(effective_conv_lr)
        wt_thetas.append(wt_theta)
        wt_conv_thetas.append(wt_conv_theta)
        w0_thetas.append(w0_theta)
        w0_conv_thetas.append(w0_conv_theta)
        wt_theta_iters.extend(wt_theta_iter_one_epoch)
        wt_conv_theta_iters.extend(wt_conv_theta_iter_one_epoch)

        logger.info('Degree from w0: {}\t Degree from w0_conv: {}'.format(w0_theta, w0_conv_theta))
        logger.info('Degree from w_t-1: {}\t Degree from w_t-1_conv: {}'.format(wt_theta, wt_conv_theta))
        if args.tensorboard:
            log_value('w0_theta', w0_theta, epoch)
            log_value('wt_theta', wt_theta, epoch)
            log_value('w0_conv_theta', w0_conv_theta, epoch)
            log_value('wt_conv_theta', wt_conv_theta, epoch)

        prev_l2_norm = wt_l2_norm
        prev_conv_l2_norm = conv_l2_norm
        prev_weight = wt_weight
        prev_conv_weight = wt_conv_weight
    
        # evaluate on validation set
        if epoch % args.test_freq == 0 or epoch == args.epochs - 1:
            acc = test(logger, test_loader, model, criterion, epoch)
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
            logger.info('Best accuracy: %f' %best_acc)
            val_errors.append(100.-acc)

            # remember best acc@1 and save checkpoint
            if args.save_model:
                save_checkpoint({
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'model': model.state_dict(),
                }, save_dir, is_best)

    logger.info('Fianal accuracy: %f' %acc)
    os.makedirs(save_dir+'/'+str(acc))
    args.lr = lr
    args.weight_decay = weight_decay
    
    # deactivate logger
    logger.removeHandler(fileHandler)
    logger.removeHandler(streamHandler)
    logging.shutdown()
    del logger, fileHandler, streamHandler 
    if args.tensorboard: unconfigure()

    utils.save_figures_epoch(save_data, save_dir, args, seed)
    return [np.array(acc), save_data]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Optimal Weight Decay Finding')
    parser.add_argument('--num_workers', default=2, type=int, 
                        help='number_workers in data_loader')
    parser.add_argument('--epochs', default=300, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--num_sample', default=0, type=int, 
                        help='number of training samples')   
    parser.add_argument('--start-epoch', default=0, type=int, 
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--start_episode', default=0, type=int, 
                        help='manual episode number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='input batch size')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, 
                        help='initial learning rate')
    parser.add_argument('--lr_end', default=0.0001, type=float, 
                        help='initial learning rate')      
    parser.add_argument('--schedule_type', default='step', type=str,
                        help='learning rate scheduling type')                            
    parser.add_argument('--decay_epoch', type=float, nargs='+', default=[150,225], 
                        help='Learning Rate Decay Steps')
    parser.add_argument('--warmup_epochs', type=int, default=0, 
                        help='Warm up epochs')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='momentum')
    parser.add_argument('--lr_decay', default=0.1, type=float, 
                        help='learning rate decay ratio')
    parser.add_argument('--nesterov', default=False, action='store_true',
                        help='If true use nesterov momentum')    
    parser.add_argument('--weight_decay', '--wd', default=0.0005, type=float, 
                        help='weight decay')
    parser.add_argument('--filter_bn_bias', default=False, action='store_true',
                        help='If true, do not give weight decay to bn parameter')
    # parser.add_argument('--filter_bias', default=False, action='store_true',
    #                     help='If true, do not give weight decay to bias')
    parser.add_argument('--net_type', default='resnet18', 
                        help='resnet18, densenetBC100')
    parser.add_argument('--zero_init_residual', default=True, action='store_false',
                        help='zero_init_residual of last residual BN')
    # parser.add_argument('--std_weight', default=1., type=float, 
    #                     help='initialization parameter. weight of std in He initialize')    
    parser.add_argument('--dataset', default='cifar10', 
                        help='cifar10 | cifar100 | svhn')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='number of classes (automatically setted in main function)')
    parser.add_argument('--dataroot', default='/home/user/dataset', 
                        help='path to dataset')
    parser.add_argument('--GCP', default=False, action='store_true', 
                        help='run on GCP?')
    parser.add_argument('--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--test_freq', default=1, type=int,
                        help='test frequency (default: 10)')
    parser.add_argument('--angle_freq', default=390, type=int,
                        help='test frequency (default: 10)')                        
    parser.add_argument('--no-augment', dest='augment', action='store_false', 
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--save_dir', type=str, default='logs/',
                        help='Directory name to save the checkpoints')    
    parser.add_argument('--save_model', default=False, action='store_true',
                        help='If true, save model every episode')
    parser.add_argument('--checkpoint', default='checkpoint.pth.tar', type=str, 
                        help='checkpoint file name')
    parser.add_argument('-l', '--load_dir', default='', type=str, 
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--no_tensorboard', dest='tensorboard', action='store_false', 
                        help='Whether to log progress to TensorBoard')
    parser.add_argument('--drop_last', default=True, action='store_false',
                        help='If true, drop last batch in data loader')
    parser.add_argument('--seed', type=int, nargs='+', default=[0,1,2,3,4], 
                        help='random seed')
    # cudnn benckmark
    parser.add_argument('--cudnn', default=True, action='store_false', 
                    help='use cudnn benchmark. Must be False using 2080ti')
    parser.set_defaults(tensorboard=True)
    # for automated mixed precision training
    parser.add_argument('--amp', default=True, action='store_false',
                        help='If true, training with mixeds precision training')
    # clipping gradient norm
    parser.add_argument('--max_grad_norm', default=None, type=float, 
                        help='clipping max gradient norm')
    parser.add_argument('--eps', default=1e-05, type=float, 
                        help='batchnorm epsilon')
    parser.add_argument('--alpha_sqaure', default=None, type=float, 
                        help='weight norm scale')    

    args = parser.parse_args()
    # if args.lr > 0.1 and args.alpha_sqaure is None:
    #     args.save_dir = 'logs_invariant_lr/'
    
    if args.GCP: 
        local_dir = '/home/user/code/jusung/weight_direction/' # -> in docker, shared_storage/code/jusung/weight_direction
    else:
        local_dir = '/home/user/jusung/weight_direction/'

    if args.GCP:
        if args.dataset == "imagenet":
            # args.dataroot = '/home/user/data/lgaivision-imagenet1k-us'
            args.dataroot = '/home/user/code/jusung/ramdisk/lgaivision-imagenet1k-us'
        else:
            args.dataroot = '/home/user/code/jusung/dataset' # CIFAR10 is saved to shared_storage/code/juseung/dataset
    else: # KAIST
        if args.dataset == "imagenet":
            args.dataroot = '/home/user/dataset/ILSVRC2012'
        elif args.dataset == "tinyimagenet":
            args.dataroot = '/home/user/dataset/tiny-imagenet-200'
        else: # CIFAR-10, CIFAR-100, STL-10
            args.dataroot = '/home/user/dataset'

    # number of dataset 
    if args.num_sample == 0:    # only cifar10
        len_dataset = 50000
    else:
        len_dataset = args.num_sample

    if args.alpha_sqaure is not None:
        args.lr = args.lr*args.alpha_sqaure
        args.weight_decay = args.weight_decay/args.alpha_sqaure
        args.eps = args.eps*args.alpha_sqaure

    args.angle_freq = int((len_dataset//args.batch_size) * (args.epochs/300))
    ####################################################################################################
    os.makedirs(local_dir+args.save_dir+args.dataset+'/'+args.net_type+'/num_data_'+str(args.num_sample)+
        '/batch_'+str(args.batch_size)+'_epoch_'+str(args.epochs), exist_ok=True)
    os.chmod(local_dir+args.save_dir, 0o777)
    os.chmod(local_dir+args.save_dir+args.dataset, 0o777)
    os.chmod(local_dir+args.save_dir+args.dataset+'/'+args.net_type, 0o777)
    os.chmod(local_dir+args.save_dir+args.dataset+'/'+args.net_type+'/num_data_'+str(args.num_sample), 0o777)
    os.chmod(local_dir+args.save_dir+args.dataset+'/'+args.net_type+'/num_data_'+str(args.num_sample)
    +'/batch_'+str(args.batch_size)+'_epoch_'+str(args.epochs), 0o777)

    # FTP location (where we save logs)
    copy_dir = '{}{}/{}/num_data_{}/batch_{}_epoch_{}/\
angle_freq_{}_{}_WD_{}_lr{}_warmup_{}_filter_bn_bias_{}_moment_{}_nester_{}_amp_{}_seed{}_'.format(
        args.save_dir, args.dataset, args.net_type, args.num_sample, args.batch_size, args.epochs, \
        args.angle_freq, args.schedule_type, args.weight_decay, args.lr, args.warmup_epochs, args.filter_bn_bias, \
        args.momentum, args.nesterov, args.amp, \
        str(args.seed).replace("[", "").replace("]", "").replace(",", "_"))
    if args.save_model:
        copy_dir = copy_dir + 'save'

    i=0
    while os.path.isdir(copy_dir + str(i)) or os.path.isdir(local_dir+copy_dir + str(i)):
        i += 1
    copy_dir = copy_dir + str(i)
    if args.load_dir: 
        copy_dir = args.load_dir

    args.save_dir = local_dir+copy_dir
    os.makedirs(args.save_dir, exist_ok=True)
    os.chmod(args.save_dir, 0o777)

###################### main ######################
    if args.amp:
        from torch.cuda.amp import GradScaler
        from torch.cuda.amp import autocast

    if args.seed is None:
        args.seed = [random.randint(0,sys.maxsize) for _ in range(5)]
    seeds_accs = [] 
    train_errors_list, val_errors_list = [],[]
    norms_list, conv_norms_list = [], []
    effective_lr_list, effective_conv_lr_list = [], []
    wt_thetas_list, wt_conv_thetas_list = [],[]
    w0_thetas_list, w0_conv_thetas_list = [],[]
    wt_theta_iters_list, wt_conv_theta_iters_list = [],[]

    for seed in args.seed:
        np.random.seed(seed)
        seeds_acc, save_data = main(args, seed)
        seeds_accs.append(seeds_acc)
        train_errors_list.append(save_data['train_errors'])
        val_errors_list.append(save_data['val_errors'])
        norms_list.append(save_data['wt_norms'])
        conv_norms_list.append(save_data['conv_norms'])
        effective_lr_list.append(save_data['effective_lrs'])
        effective_conv_lr_list.append(save_data['effective_conv_lrs'])
        wt_thetas_list.append(save_data['wt_thetas'])
        wt_conv_thetas_list.append(save_data['wt_conv_thetas'])
        w0_thetas_list.append(save_data['w0_thetas'])
        w0_conv_thetas_list.append(save_data['w0_conv_thetas'])
        wt_theta_iters_list.append(save_data['wt_theta_iters'])
        wt_conv_theta_iters_list.append(save_data['wt_conv_theta_iters'])

    # save accuracy log
    average_acc = np.average(seeds_accs,0)
    std_acc = np.std(seeds_accs,0)
    seeds_accs.append(average_acc)
    seeds_accs.append(std_acc)
    args.seed.append('average')
    args.seed.append('std')
    avg_acc_file = pd.DataFrame(seeds_accs, \
        columns=np.arange(1), index=args.seed)
    avg_acc_file.to_excel(args.save_dir+'/avg_acc_file.xlsx')
    os.makedirs(args.save_dir+'/'+str(average_acc).replace("[", "").replace("]", "").replace(",", " "))

    # save effective lr log
    average_effective_lr = np.average(effective_lr_list,0)
    std_effective_lr = np.std(effective_lr_list,0)
    effective_lr_list.append(average_effective_lr)
    effective_lr_list.append(std_effective_lr)

    # save effective conv lr log
    average_effective_conv_lr = np.average(effective_conv_lr_list,0)
    std_effective_conv_lr = np.std(effective_conv_lr_list,0)
    effective_conv_lr_list.append(average_effective_conv_lr)
    effective_conv_lr_list.append(std_effective_conv_lr)

    # save train error log
    average_train_error = np.average(train_errors_list,0)
    std_train_error = np.std(train_errors_list,0)
    train_errors_list.append(average_train_error)
    train_errors_list.append(std_train_error)
    # save val error log
    average_val_error = np.average(val_errors_list,0)
    std_val_error = np.std(val_errors_list,0)
    val_errors_list.append(average_val_error)
    val_errors_list.append(std_val_error)
    # save norm log
    average_norm = np.average(norms_list,0)
    std_norm = np.std(norms_list,0)
    norms_list.append(average_norm)
    norms_list.append(std_norm)
    # save conv_norm log
    average_conv_norm = np.average(conv_norms_list,0)
    std_conv_norm = np.std(conv_norms_list,0)
    conv_norms_list.append(average_conv_norm)
    conv_norms_list.append(std_conv_norm)
    # save theta log
    average_theta = np.average(wt_thetas_list,0)
    std_theta = np.std(wt_thetas_list,0)
    wt_thetas_list.append(average_theta)
    wt_thetas_list.append(std_theta)
    # save conv_theta log
    average_conv_theta = np.average(wt_conv_thetas_list,0)
    std_conv_theta = np.std(wt_conv_thetas_list,0)
    wt_conv_thetas_list.append(average_conv_theta)
    wt_conv_thetas_list.append(std_conv_theta)
    # save theta_w0 log
    average_theta0 = np.average(w0_thetas_list,0)
    std_theta0 = np.std(w0_thetas_list,0)
    w0_thetas_list.append(average_theta0)
    w0_thetas_list.append(std_theta0)
    # save conv_theta log
    average_conv_theta0 = np.average(w0_conv_thetas_list,0)
    std_conv_theta0 = np.std(w0_conv_thetas_list,0)
    w0_conv_thetas_list.append(average_conv_theta0)
    w0_conv_thetas_list.append(std_conv_theta0)
    # save conv_theta_iters log
    average_wt_theta_iters = np.average(wt_theta_iters_list,0)
    std_wt_theta_iters = np.std(wt_theta_iters_list,0)
    wt_theta_iters_list.append(average_wt_theta_iters)
    wt_theta_iters_list.append(std_wt_theta_iters)
    # save conv_theta_iters log
    average_wt_conv_theta_iters = np.average(wt_conv_theta_iters_list,0)
    std_wt_conv_theta_iters = np.std(wt_conv_theta_iters_list,0)
    wt_conv_theta_iters_list.append(average_wt_conv_theta_iters)
    wt_conv_theta_iters_list.append(std_wt_conv_theta_iters)

    name= ['train_error']*len(train_errors_list)+['val_error']*len(val_errors_list)+\
        ['norm']*len(norms_list)+['conv_norm']*len(conv_norms_list)+\
        ['wt_theta']*len(wt_thetas_list)+['wt_conv_theta']*len(wt_conv_thetas_list)+\
        ['w0_theta']*len(w0_thetas_list)+['w0_conv_theta']*len(w0_conv_thetas_list)+\
        ['effective_lr']*len(effective_lr_list)+['effective_conv_lr']*len(effective_conv_lr_list)
    name2 = ['wt_theta_iter']*len(wt_theta_iters_list)+['wt_conv_theta_iter']*len(wt_conv_theta_iters_list)
    excel_data, excel_data2 = [], []
    excel_data.extend(train_errors_list)
    excel_data.extend(val_errors_list)
    excel_data.extend(norms_list)
    excel_data.extend(conv_norms_list)
    excel_data.extend(wt_thetas_list)
    excel_data.extend(wt_conv_thetas_list)
    excel_data.extend(w0_thetas_list)
    excel_data.extend(w0_conv_thetas_list)
    excel_data.extend(effective_lr_list)
    excel_data.extend(effective_conv_lr_list)
    excel_data2.extend(wt_theta_iters_list)
    excel_data2.extend(wt_conv_theta_iters_list)
    avg_norm_file = pd.DataFrame(excel_data, columns=np.arange(args.epochs+1), index=[name, args.seed*10])
    avg_norm_file.to_excel(args.save_dir+'/direction_file_{}_{}_{}_{}.xlsx'.format(
        args.num_sample, int(args.batch_size), args.lr, args.weight_decay))
    angle_iter_file = pd.DataFrame(excel_data2, columns=np.arange(args.epochs*(len_dataset//args.batch_size)//args.angle_freq), 
        index=[name2, args.seed*2])  
    angle_iter_file.to_excel(args.save_dir+'/angle_file_{}_{}_{}_{}_{}.xlsx'.format(
        args.epochs, args.num_sample, int(args.batch_size), args.lr, args.weight_decay))
    
    # save figure
    # plt.errorbar(np.arange(args.epochs+1), average_norm, std_norm)
    sns.set_theme(style="darkgrid")

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), average_train_error, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs), average_train_error-std_train_error, average_train_error+std_train_error, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('training error %', fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/training_error.pdf'.format(args.save_dir), dpi=300)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), average_val_error, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs), average_val_error-std_val_error, average_val_error+std_val_error, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('validation error %', fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/validation_error.pdf'.format(args.save_dir), dpi=300)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs+1), average_norm, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs+1), average_norm-std_norm, average_norm+std_norm, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('L2 norm', fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/L2_norm.pdf'.format(args.save_dir), dpi=300)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs+1), average_conv_norm, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs+1), average_conv_norm-std_conv_norm, average_conv_norm+std_conv_norm, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('L2 norm', fontsize=18)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/L2_conv_norm.pdf'.format(args.save_dir), dpi=300)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), average_theta, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs), average_theta-std_theta, average_theta+std_theta, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/theta.pdf'.format(args.save_dir), dpi=300)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), average_conv_theta, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs), average_conv_theta-std_conv_theta, average_conv_theta+std_conv_theta, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/conv_theta.pdf'.format(args.save_dir), dpi=300)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), average_theta0, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs), average_theta0-std_theta0, average_theta0+std_theta0, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/theta_w0.pdf'.format(args.save_dir), dpi=300)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), average_conv_theta0, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs), average_conv_theta0-std_conv_theta0, average_conv_theta0+std_conv_theta0, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/conv_theta_w0.pdf'.format(args.save_dir), dpi=300)

    # effective lr
    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), average_effective_lr, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs), average_effective_lr-std_effective_lr, average_effective_lr+std_effective_lr, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('effecive learning rate', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.18)
    fig.savefig('{}/effective_lr.pdf'.format(args.save_dir), dpi=300)

    # effective conv_lr
    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs), average_effective_conv_lr, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs), 
            average_effective_conv_lr-std_effective_conv_lr, average_effective_conv_lr+std_effective_conv_lr, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('effecive learning rate', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.18)
    fig.savefig('{}/effective_conv_lr.pdf'.format(args.save_dir), dpi=300)

    # weight polar plot
    # plt.clf()
    # fig = plt.figure(1, figsize=(8, 4))
    fig = plt.figure()
    # fig = plt.subplots()
    # fig.subplots_adjust(wspace=0.2, left=0.2, right=0.8)
    
    ax, aux_ax = utils.setup_axes(fig, 111, theta=[0, 90], radius=[0, max(average_conv_norm)*1.1])
    # generate the data to plot
    theta=np.array([0])
    theta=np.concatenate([theta,average_theta0]) # in degrees
    radius = average_conv_norm
    aux_ax.plot(theta, radius)
    # plt.tight_layout()
    fig.set_size_inches(3.75,3.75)
    fig.savefig('{}/weight_polar.pdf'.format(args.save_dir), dpi=300) 

    # theta_iters
    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs*((len_dataset//args.batch_size))//args.angle_freq), average_wt_theta_iters, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs*((len_dataset//args.batch_size))//args.angle_freq), 
                    average_wt_theta_iters-std_wt_theta_iters, average_wt_theta_iters+std_wt_theta_iters, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/theta_iter.pdf'.format(args.save_dir), dpi=300)

    fig, ax = plt.subplots()
    image, = ax.plot(np.arange(args.epochs*((len_dataset//args.batch_size))//args.angle_freq), average_wt_conv_theta_iters, linewidth=3, alpha=0.9)
    ax.fill_between(np.arange(args.epochs*((len_dataset//args.batch_size))//args.angle_freq), 
                    average_wt_conv_theta_iters-std_wt_conv_theta_iters, average_wt_conv_theta_iters+std_wt_conv_theta_iters, alpha=0.2)
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('degree', fontsize=18)    
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.gcf().subplots_adjust(left=0.15)
    fig.savefig('{}/conv_theta_iter.pdf'.format(args.save_dir), dpi=300)


    if not args.GCP:
        try:
            shutil.copytree(args.save_dir, copy_dir)
            os.chmod(copy_dir, 0o777)
        except:
            shutil.copytree(args.save_dir, copy_dir+'_copy2')
            os.chmod(copy_dir+'_copy2', 0o777)
    