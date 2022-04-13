'''
******  Test whether the network is invariant or not  ******
'''
from __future__ import print_function
# from __future__ import division

import torch
# torch.backends.cudnn.deterministic = True    # for compare
import torch.nn as nn
import torch.optim as optim
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
from datetime import datetime
# import tensorboard_logger
from tensorboard_logger.tensorboard_logger import configure, unconfigure, log_value
# from tensorboard_logger import configure, log_value
import models
import data_loader
import active_util
# import sgd_js
import pdb
import copy

# Training
def train(logger, train_loader, model1, model2, criterion, optimizer1, optimizer2, epoch, num_iter_for_update):
    '''
    model1: learning rate scaling
    model2: weight decay rate scaling    
    '''

    batch_time = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()
    top11 = AverageMeter()
    top12 = AverageMeter()

    model1.eval()
    model2.eval()
    end = time.time()

    if args.amp:
        scaler1 = GradScaler()
        scaler2 = GradScaler()
    optimizer1.zero_grad()
    optimizer2.zero_grad()
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs, targets.cuda()

        # compute output
        if args.amp:
            with autocast():
                outputs1 = model1(inputs)
                outputs2 = model2(inputs)
                loss1 = criterion(outputs1, targets) / num_iter_for_update
                loss2 = criterion(outputs2, targets) / num_iter_for_update
        else:
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            loss1 = criterion(outputs1, targets) / num_iter_for_update
            loss2 = criterion(outputs2, targets) / num_iter_for_update

        # print('\t{}'.format(outputs1[0]))
        # print('\t{}'.format(outputs2[0]))
        # print(model1.module.conv1.weight[0][0])
        # print(model2.module.conv1.weight[0][0])

        # measure accuracy and record loss
        acc11 = accuracy(outputs1.data, targets, topk=(1,))[0]
        acc12 = accuracy(outputs2.data, targets, topk=(1,))[0]
        losses1.update(loss1.data, inputs.size(0))
        losses2.update(loss2.data, inputs.size(0))
        top11.update(acc11, inputs.size(0))
        top12.update(acc12, inputs.size(0))

        # compute gradient and do SGD step
        if args.amp:
            scaler1.scale(loss1).backward()
            scaler2.scale(loss2).backward()
            if (i+1) % num_iter_for_update == 0:
                scaler1.step(optimizer1)
                scaler2.step(optimizer2)
                scaler1.update()
                scaler2.update()
                if i+1 != len(train_loader):
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
        else:
            loss1.backward()
            loss2.backward()
            if (i+1) % num_iter_for_update == 0 :
                optimizer1.step()
                optimizer2.step()
                if i+1 != len(train_loader):
                    optimizer1.zero_grad()
                    optimizer2.zero_grad()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss1 {loss1.val:.4f} Loss2 {loss2.val:.4f}\t'
                        'acc1 {top11.val:.3f} acc2 {top12.val:.3f}'.format(
                            epoch, i, len(train_loader), batch_time=batch_time,
                            loss1=losses1, loss2=losses2, top11=top11, top12=top12))

    # log to TensorBoard
    if args.tensorboard:
        log_value('train_loss1', losses1.avg, epoch)
        log_value('train_loss2', losses2.avg, epoch)
        log_value('train_acc1', top11.avg, epoch)
        log_value('train_acc2', top12.avg, epoch)

        # compute norm
        model1.eval()
        model2.eval()
        weight_l2_norm1, grad_l2_norm1, conv_l2_norm1, conv_grad_l2_norm1 = 0,0,0,0
        weight_l2_norm2, grad_l2_norm2, conv_l2_norm2, conv_grad_l2_norm2 = 0,0,0,0
        wt_weight1, wt_weight2 = [],[]
        wt_conv_weight1, wt_conv_weight2 = [],[]
        if 'resnet' in args.net_type:
            conv_parameters = ['conv', 'shortcut.0']
        elif 'densenet' in args.net_type:
            conv_parameters = ['conv']
        with torch.no_grad():
            for name, param in model1.named_parameters():
                if not 'linear' in name:
                    weight_l2_norm1 = weight_l2_norm1 + (param.data.norm(2).item()**2)
                    grad_l2_norm1 = grad_l2_norm1 + param.grad.data.norm(2).item()**2
                    wt_weight1.append(param.view(-1).data)
                    
                if any(conv_parameter in name for conv_parameter in conv_parameters):
                    conv_l2_norm1 = conv_l2_norm1 + param.data.norm(2).item()**2
                    conv_grad_l2_norm1 = conv_grad_l2_norm1 + param.grad.data.norm(2).item()**2
                    wt_conv_weight1.append(param.view(-1).data)
            wt_weight1 = torch.cat(wt_weight1,0)
            wt_conv_weight1 = torch.cat(wt_conv_weight1,0)
            weight_l2_norm1 = weight_l2_norm1**(1./2)
            conv_l2_norm1 = conv_l2_norm1**(1./2)
            conv_grad_l2_norm1 = conv_grad_l2_norm1**(1./2)

            for name, param in model2.named_parameters():
                if not 'linear' in name:
                    weight_l2_norm2 = weight_l2_norm2 + (param.data.norm(2).item()**2)
                    grad_l2_norm2 = grad_l2_norm2 + param.grad.data.norm(2).item()**2
                    wt_weight2.append(param.view(-1).data)
                    
                if any(conv_parameter in name for conv_parameter in conv_parameters):
                    conv_l2_norm2 = conv_l2_norm2 + param.data.norm(2).item()**2
                    conv_grad_l2_norm2 = conv_grad_l2_norm2 + param.grad.data.norm(2).item()**2
                    wt_conv_weight2.append(param.view(-1).data)
            wt_weight2 = torch.cat(wt_weight2,0)
            wt_conv_weight2 = torch.cat(wt_conv_weight2,0)
            weight_l2_norm2 = weight_l2_norm2**(1./2)
            conv_l2_norm2 = conv_l2_norm2**(1./2)
            conv_grad_l2_norm2 = conv_grad_l2_norm2**(1./2)

        logger.info('conv_l2_norm ratio {}'.format(conv_l2_norm1/conv_l2_norm2))
        logger.info('w_l2_norm ratio {}'.format(weight_l2_norm1/weight_l2_norm2))

        log_value('weight_l2_norm1', weight_l2_norm1, epoch+1)
        log_value('weight_l2_norm2', weight_l2_norm2, epoch+1)
        log_value('grad_l2_norm1', grad_l2_norm1**(1./2), epoch)
        log_value('grad_l2_norm2', grad_l2_norm2**(1./2), epoch)
        log_value('conv_l2_norm1', conv_l2_norm1, epoch+1)
        log_value('conv_l2_norm2', conv_l2_norm2, epoch+1)
        log_value('conv_grad_l2_norm1', conv_grad_l2_norm1, epoch)
        log_value('conv_grad_l2_norm2', conv_grad_l2_norm2, epoch)

        return [weight_l2_norm1, weight_l2_norm2, conv_l2_norm1, conv_l2_norm2, wt_weight1, wt_weight2, wt_conv_weight1, wt_conv_weight2]
          

def test(logger, test_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    confusion_matrix = torch.zeros(args.num_classes, args.num_classes)
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
            
            # compute class separate accuracy
            for t, p in zip(targets.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

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

        accuracy_each_class = confusion_matrix.diag()/confusion_matrix.sum(1)
        logger.info(' * acc@1 {top1.avg:.3f}'.format(top1=top1))
        logger.info(' * accuracy of each classes: \n  {}'.format(accuracy_each_class))

    # log to TensorBoard
    if args.tensorboard:
        log_value('val_loss', losses.avg, epoch)
        log_value('val_acc', top1.avg, epoch)

    return top1.avg


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

    if epoch in args.epoch_step:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.lr_decay
            lr = param_group['lr']
            args.lr = lr
    elif epoch==0:
        lr = 0            
    else:
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

    # log to TensorBoard
    if args.tensorboard:
        log_value('learning_rate', lr, epoch)
        log_value('weight_decay1', args.weight_decay1, epoch)
        log_value('weight_decay2', args.weight_decay2, epoch)


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
    if 'resnet' in args.net_type:
        bn_parameters = ['bn', 'shortcut.1']
    elif 'densenet' in args.net_type:
        bn_parameters = ['bn']

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        # print('{}\t shape: {}'.format(name, param.shape))
        if not param.requires_grad:
            continue
        if 'linear' in name:
            # do not give weight decay to linear layer
            no_decay.append(param)
        elif args.filter_bn_bias:
            # do not give weight decay to BN and bias
            if len(param.shape) == 1 or name in skip_list:
                no_decay.append(param)
            else:
                decay.append(param)
        else:
             # give weight decay to all param except for linear
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

def main(args, seed):
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
    elif args.dataset == 'imagenet':
        args.num_classes = 1000

    if args.dataset == "imagenet" and args.batch_size >= 512:
        num_iter_for_update = args.batch_size / 512
    elif 'resnet' in args.net_type and 'cifar' in args.dataset and args.batch_size > 16384:
        num_iter_for_update = args.batch_size / 16384
    elif 'densenet' in args.net_type and 'cifar' in args.dataset and args.batch_size > 8192:
        num_iter_for_update = args.batch_size / 8192
    else:
        num_iter_for_update = 1
    loader_batch_size = int(args.batch_size/num_iter_for_update)

    if args.load_dir:
        if os.path.isdir(args.load_dir):
            # Load checkpoint.
            print("=> loading checkpoint '{}'".format(args.load_dir))
            checkpoint = torch.load(args.load_dir+'/'+args.checkpoint)
        else:
            print("=> no checkpoint found at '{}'".format(args.load_dir))

    # Data
    print('==> Preparing data..')
    if args.num_sample is not None:
        labeled_idx = np.random.choice(50000, args.num_sample, replace=False)
        sampler = torch.utils.data.sampler.SubsetRandomSampler(labeled_idx)
        train_loader, test_loader = data_loader.getDataSet(args.dataset+'_eval', loader_batch_size, args.dataroot, \
                sampler=sampler, drop_last=args.drop_last, num_workers=args.num_workers)
    else:
        train_loader, test_loader = data_loader.getDataSet(args.dataset, loader_batch_size, args.dataroot, \
                drop_last=args.drop_last, num_workers=args.num_workers)

    # Model
    print('==> Building model..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_to_call = getattr(models, args.net_type)
    model1 = model_to_call(num_classes=args.num_classes, zero_init_residual=args.zero_init_residual)
    model1 = model1.to(device)
    model1 = torch.nn.DataParallel(model1)
    # if device == 'cuda' and args.cudnn:
    #     cudnn.benchmark = True         # for compare
    # cudnn.benchmark = False
    model2 = copy.deepcopy(model1)

    criterion = nn.CrossEntropyLoss()
    lr1 = args.lr1
    lr2 = args.lr2
    weight_decay1 = args.weight_decay1
    weight_decay2 = args.weight_decay2

    # if weight_decay and args.filter_bn:
    parameters1 = add_weight_decay(model1, weight_decay1)
    parameters2 = add_weight_decay(model2, weight_decay2)
    # else:
    #     parameters = model.parameters()

    optimizer1 = optim.SGD(parameters1, lr=args.lr1, momentum=args.momentum, weight_decay=args.weight_decay1, nesterov=args.nesterov)
    optimizer2 = optim.SGD(parameters2, lr=args.lr2, momentum=args.momentum, weight_decay=args.weight_decay2, nesterov=args.nesterov)

    # episode code copy
    best_acc = 0
    conv_norms1, conv_norms2 = [], []                   # for save norm excel file
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer1, epoch)
        adjust_learning_rate(optimizer2, epoch)

        # save norm of w0
        if epoch == 0:
            w0_l2_norm1, w0_l2_norm2 = 0,0      # norm of w0_all (without linear)
            w0_conv_l2_norm1, w0_conv_l2_norm2 = 0,0    # norm of w0_conv
            w0_weight1, w0_weight2 = [],[]              # weight vector to calculate directional update
            w0_conv_weight1, w0_conv_weight2 = [],[]    # weight vector to calculate directional update
            if 'resnet' in args.net_type:
                conv_parameters = ['conv', 'shortcut.0']
                bn_parameters = ['bn', 'shortcut.1']
            elif 'densenet' in args.net_type:
                conv_parameters = ['conv']
                bn_parameters = ['bn']
                    
            with torch.no_grad():
                # model1
                for name, param in model1.named_parameters():
                    # make the network scale invariant
                    if 'linear' in name or 'classifier' in name:
                        param.requires_grad = False
                    else:
                        w0_l2_norm1 = w0_l2_norm1 + (param.data.norm(2).item()**2)
                        w0_weight1.append(param.view(-1).data)
                    if any(conv_parameter in name for conv_parameter in conv_parameters):
                        w0_conv_l2_norm1 = w0_conv_l2_norm1 + param.data.norm(2).item()**2
                        w0_conv_weight1.append(param.view(-1).data)

                # model2
                for name, param in model2.named_parameters():
                    # make the network scale invariant
                    if 'linear' in name or 'classifier' in name:
                        param.requires_grad = False
                    else:
                        if args.base_norm is not None:
                            param.mul_(np.sqrt(args.base_norm/args.batch_size))
                        w0_l2_norm2 = w0_l2_norm2 + (param.data.norm(2).item()**2)
                        w0_weight2.append(param.view(-1).data)

                    if any(conv_parameter in name for conv_parameter in conv_parameters):
                        w0_conv_l2_norm2 = w0_conv_l2_norm2 + param.data.norm(2).item()**2
                        w0_conv_weight2.append(param.view(-1).data)

                w0_weight1 = torch.cat(w0_weight1,0)   
                w0_weight2 = torch.cat(w0_weight2,0)
                w0_conv_weight1 = torch.cat(w0_conv_weight1,0)
                w0_conv_weight2 = torch.cat(w0_conv_weight2,0)
                w0_conv_l2_norm1 = w0_conv_l2_norm1**(1./2)
                w0_conv_l2_norm2 = w0_conv_l2_norm2**(1./2)
                w0_l2_norm1 = w0_l2_norm1**(1./2)
                w0_l2_norm2 = w0_l2_norm2**(1./2)
                
                # previous norm
                prev_l2_norm1 = w0_l2_norm1
                prev_l2_norm2 = w0_l2_norm2
                prev_conv_l2_norm1 = w0_conv_l2_norm1
                prev_conv_l2_norm2 = w0_conv_l2_norm2
                # previous weight
                prev_weight1 = w0_weight1
                prev_weight2 = w0_weight2
                prev_conv_weight1 = w0_conv_weight1
                prev_conv_weight2 = w0_conv_weight2

                # for save norm excel file
                conv_norms1.append(w0_conv_l2_norm1)
                conv_norms2.append(w0_conv_l2_norm2)
                ####################################################
                ####################################################


            logger.info('w_l2_norm ratio {}'.format(w0_l2_norm1/w0_l2_norm2))
            logger.info('conv_l2_norm ratio {}'.format(w0_conv_l2_norm1/w0_conv_l2_norm2))
            if args.tensorboard:
                log_value('weight_l2_norm1', w0_l2_norm1, epoch)
                log_value('weight_l2_norm2', w0_l2_norm2, epoch)
                log_value('conv_l2_norm1', w0_conv_l2_norm1, epoch)
                log_value('conv_l2_norm2', w0_conv_l2_norm2, epoch)

        # train for one epoch
        weight_l2_norm1, weight_l2_norm2, conv_l2_norm1, conv_l2_norm2, wt_weight1, wt_weight2, wt_conv_weight1, wt_conv_weight2 = train(
                                        logger, train_loader, model1, model2, criterion, optimizer1, optimizer2, epoch, num_iter_for_update)
        conv_norms1.append(conv_l2_norm1)
        conv_norms2.append(conv_l2_norm2)
        ####################################################
        ####################################################
        
        # calculate angular update in the degree
        w0_theta1 = torch.acos(torch.dot(w0_weight1, wt_weight1)/(w0_l2_norm1*weight_l2_norm1)).cpu()*(360./(2.*np.pi))
        w0_theta2 = torch.acos(torch.dot(w0_weight2, wt_weight2)/(w0_l2_norm2*weight_l2_norm2)).cpu()*(360./(2.*np.pi))
        w0_conv_theta1 = torch.acos(torch.dot(w0_conv_weight1, wt_conv_weight1)/(w0_conv_l2_norm1*conv_l2_norm1)).cpu()*(360./(2.*np.pi))
        w0_conv_theta2 = torch.acos(torch.dot(w0_conv_weight2, wt_conv_weight2)/(w0_conv_l2_norm2*conv_l2_norm2)).cpu()*(360./(2.*np.pi))

        wt_cos_theta1 = torch.clamp(torch.dot(prev_weight1, wt_weight1)/(prev_l2_norm1*weight_l2_norm1),-1,1).cpu() # clamp for stability
        wt_cos_theta2 = torch.clamp(torch.dot(prev_weight2, wt_weight2)/(prev_l2_norm2*weight_l2_norm2),-1,1).cpu() # clamp for stability
        wt_conv_cos_theta1 = torch.clamp(torch.dot(prev_conv_weight1, wt_conv_weight1)/(prev_conv_l2_norm1*conv_l2_norm1),-1,1).cpu() # clamp for stability
        wt_conv_cos_theta2 = torch.clamp(torch.dot(prev_conv_weight2, wt_conv_weight2)/(prev_conv_l2_norm2*conv_l2_norm2),-1,1).cpu() # clamp for stability

        wt_theta1 = torch.acos(wt_cos_theta1)*(360./(2.*np.pi))
        wt_theta2 = torch.acos(wt_cos_theta2)*(360./(2.*np.pi))
        wt_conv_theta1 = torch.acos(wt_conv_cos_theta1)*(360./(2.*np.pi))
        wt_conv_theta2 = torch.acos(wt_conv_cos_theta2)*(360./(2.*np.pi)) 

        print('Degree from w0: {}\t {}'.format(w0_theta1, w0_theta2))
        print('Degree from w0_conv: {}\t {}'.format(w0_conv_theta1, w0_conv_theta2))
        print('Degree from t-1: {}\t {}'.format(wt_theta1, wt_theta2))
        print('Degree from t-1_conv: {}\t {}'.format(wt_conv_cos_theta1, wt_conv_cos_theta2))

        prev_l2_norm1 = weight_l2_norm1
        prev_l2_norm2 = weight_l2_norm2
        prev_conv_l2_norm1 = conv_l2_norm1
        prev_conv_l2_norm2 = conv_l2_norm2
        prev_weight1 = wt_weight1
        prev_weight2 = wt_weight2
        prev_conv_weight1 = wt_conv_weight1
        prev_conv_weight2 = wt_conv_weight2
    
        if args.tensorboard:
            log_value('Degree from w0_1', w0_theta1, epoch)
            log_value('Degree from w0_2', w0_theta2, epoch)
            log_value('Degree from t-1_1', wt_theta1, epoch)
            log_value('Degree from t-1_2', wt_theta2, epoch)
                
        # evaluate on validation set
        if epoch % args.test_freq == 0 or epoch == args.epochs - 1:
            acc = test(logger, test_loader, model1, criterion, epoch)
            is_best = acc > best_acc
            if is_best:
                best_acc = acc
            logger.info('Best accuracy: %f' %best_acc)  

            # remember best acc@1 and save checkpoint
            if args.save_model:
                save_checkpoint({
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'model': model1.state_dict(),
                }, save_dir, is_best)

    logger.info('Fianal accuracy: %f' %acc)
    os.makedirs(save_dir+'/'+str(acc.item()))
    args.lr1 = lr1
    args.lr2 = lr2
    args.weight_decay1 = weight_decay1
    args.weight_decay2 = weight_decay2
    
    logger.removeHandler(fileHandler)
    logger.removeHandler(streamHandler)
    logging.shutdown()
    del logger, fileHandler, streamHandler 
    if args.tensorboard: unconfigure()
    return [np.array(acc), np.array(conv_norms1), np.array(conv_norms2)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='PyTorch Optimal Weight Decay Finding')
    parser.add_argument('--num_workers', default=2, type=int, 
                        help='number_workers in data_loader')
    parser.add_argument('--epochs', default=300, type=int, 
                        help='number of total epochs to run')
    parser.add_argument('--num_sample', default=None, type=int, 
                        help='number of training samples')   
    parser.add_argument('--start-epoch', default=0, type=int, 
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--start_episode', default=0, type=int, 
                        help='manual episode number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=128, 
                        help='input batch size')
    parser.add_argument('--lr1', default=0.1, type=float, 
                        help='initial learning rate')
    parser.add_argument('--lr2', default=0.1, type=float, 
                        help='initial learning rate')
    parser.add_argument('--epoch_step', type=int, nargs='+', default=[150,225], 
                        help='Learning Rate Decay Steps')
    parser.add_argument('--warm_up_epoch', type=int, default=0, 
                        help='Warm up epochs')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='momentum')
    parser.add_argument('--lr_decay', default=0.1, type=float, 
                        help='learning rate decay ratio')
    parser.add_argument('--nesterov', default=False, action='store_true',
                        help='If true use nesterov momentum')    
    parser.add_argument('--weight_decay1', '--wd1', default=0.0005, type=float, 
                        help='weight decay')
    parser.add_argument('--weight_decay2', '--wd2', default=0.0005, type=float, 
                        help='weight decay')    
    parser.add_argument('--filter_bn_bias', default=False, action='store_true',
                        help='If true, do not give weight decay to bn parameter')
    # parser.add_argument('--filter_bias', default=False, action='store_true',
    #                     help='If true, do not give weight decay to bias')
    parser.add_argument('--net_type', default='resnet18', 
                        help='resnet18, densenetBC100')
    parser.add_argument('--zero_init_residual', default=False, action='store_true',
                        help='zero_init_residual of last residual BN')
    parser.add_argument('--std_weight', default=1., type=float, 
                        help='initialization parameter. weight of std in He initialize')    
    parser.add_argument('--dataset', default='cifar10', 
                        help='cifar10 | cifar100 | svhn')
    parser.add_argument('--num_classes', default=10, type=int,
                        help='number of classes (automatically setted in main function)')
    parser.add_argument('--dataroot', default='/home/user/jusung/data', 
                        help='path to dataset')
    parser.add_argument('--print-freq', default=1, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--test_freq', default=1, type=int,
                        help='test frequency (default: 10)')
    parser.add_argument('--no-augment', dest='augment', action='store_false', 
                        help='whether to use standard augmentation (default: True)')
    parser.add_argument('--save_dir', type=str, default='./logs_invariant_test/',
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
    # plot argument
    parser.add_argument('--xlim', type=float, default=None, 
                        help='xlim for plotting weight distribution')
    parser.add_argument('--ylim', type=int, default=None, 
                        help='ylim for plotting weight distribution')
    parser.add_argument('--save_fig', default=False, action='store_true',
                        help='If true, save weight distribution pdf and L2 norm of each layer')
    # cudnn benckmark
    parser.add_argument('--cudnn', default=True, action='store_false', 
                    help='use cudnn benchmark. Must be False using 2080ti')
    parser.set_defaults(tensorboard=True)
    # for automated mixed precision training
    parser.add_argument('--amp', default=True, action='store_false',
                        help='If true, training with mixed precision training')
    # clipping gradient norm
    parser.add_argument('--max_grad_norm', default=None, type=float, 
                        help='clipping max gradient norm')
    parser.add_argument('--norm_grad_ratio', default=None, type=float, 
                        help='clipping max gradient norm by the ration of weight norm')
    parser.add_argument('--base_norm', default=None, type=int, 
                        help='sqrt(base_norm/batch_size) is multiplied to weight norm')

    args = parser.parse_args()

    ####################################################################################################

    os.makedirs(args.save_dir+args.dataset+'/'+args.net_type+'/batch_'+str(args.batch_size), exist_ok=True)
    os.chmod(args.save_dir, 0o777)
    os.chmod(args.save_dir+args.dataset, 0o777)
    os.chmod(args.save_dir+args.dataset+'/'+args.net_type, 0o777)        
    os.chmod(args.save_dir+args.dataset+'/'+args.net_type+'/batch_'+str(args.batch_size), 0o777)

    args.save_dir = '{}{}/{}/batch_{}/invariant_base_norm_{}_zero_init_{}_warm_iter_{}_filter_bn_bias_{}_WD_{},{}_lr{},{}_momentum{}_nester_{}_epoch{}_amp_{}_seed{}_'.format(
        args.save_dir, args.dataset, args.net_type, args.batch_size, \
        args.base_norm, args.zero_init_residual, args.warm_up_epoch, args.filter_bn_bias, \
        args.weight_decay1, args.weight_decay2, args.lr1, args.lr2, args.momentum, args.nesterov, args.epochs, args.amp, \
        str(args.seed).replace("[", "").replace("]", "").replace(",", "_"))
    if args.save_model:
        args.save_dir = args.save_dir + 'save'

    i=0
    while os.path.isdir(args.save_dir + str(i)):
        i += 1
    args.save_dir = args.save_dir + str(i)
    if args.load_dir: 
         args.save_dir = args.load_dir
    os.makedirs(args.save_dir, exist_ok=True)
    os.chmod(args.save_dir, 0o777)

###################### main ######################
    if args.amp:
        from torch.cuda.amp import GradScaler
        from torch.cuda.amp import autocast

    if args.seed is None:
        args.seed = [random.randint(0,sys.maxsize) for _ in range(5)]
    seeds_accs, conv_norms_list1, conv_norms_list2 = [], [], []
    for seed in args.seed:
        # np.random.seed(seed)       

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0) # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(0)
        random.seed(0)

        seeds_acc, conv_norms1, conv_norms2 = main(args, seed)
        seeds_accs.append(seeds_acc)
        conv_norms_list1.append(conv_norms1)
        conv_norms_list2.append(conv_norms2)

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

    # save conv_norm log
    average_norm1 = np.average(conv_norms_list1,0)
    average_norm2 = np.average(conv_norms_list2,0)
    std_norm1 = np.std(conv_norms_list1,0)
    std_norm2 = np.std(conv_norms_list2,0)
    conv_norms_list1.append(average_norm1)
    conv_norms_list2.append(average_norm2)
    conv_norms_list1.append(std_norm1)
    conv_norms_list2.append(std_norm2)

    avg_norm_file = pd.DataFrame([conv_norms_list1, conv_norms_list2], \
        columns=np.arange(args.epochs+1), index=[args.seed, args.seed])
    avg_norm_file.to_excel(args.save_dir+'/avg_norm_file.xlsx')
    # save figure of norms
    # plt.errorbar(np.arange(args.epochs+1), average_norm, std_norm)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    # ax.set_title(title='Norm of convolution weights', xlabel='epochs', ylabel='L2 norm')
    ax.set_xlabel('epochs', fontsize=18)
    ax.set_ylabel('L2 norm', fontsize=18)

    ax.plot(np.arange(args.epochs+1), average_norm)

    plt.savefig('{}/conv_norms.pdf'.format(args.save_dir), dpi=300)