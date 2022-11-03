import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision

from coco_utils import get_coco
import presets
import utils

# JS
import logging
import shutil
from tensorboard_logger.tensorboard_logger import configure, log_value


def get_dataset(dir_path, name, image_set, transform, **kwargs):
    def sbd(*args, **kwargs):
        return torchvision.datasets.SBDataset(*args, mode='segmentation', **kwargs)
    paths = {
        "voc": (dir_path, torchvision.datasets.VOCSegmentation, 21),
        "voc_aug": (dir_path, sbd, 21),
        "coco": (dir_path, get_coco, 21)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform, **kwargs)
    return ds, num_classes


def get_transform(train):
    # base_size = 520
    # crop_size = 480
    base_size = 260
    crop_size = 240

    return presets.SegmentationPresetTrain(base_size, crop_size) if train else presets.SegmentationPresetEval(base_size)


def criterion(inputs, target):
    losses = {}
    for name, x in inputs.items():
        losses[name] = nn.functional.cross_entropy(x, target, ignore_index=255)

    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']


def evaluate(model, data_loader, device, num_classes, logger):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ", logger=logger)
    header = 'Test:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()

    return confmat


def train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq, logger, schedule_type):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ", logger=logger)
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = 'Epoch: [{}]'.format(epoch)
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)
        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if schedule_type == 'exp':
            lr_scheduler.step()

        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
    if schedule_type == 'step':
        lr_scheduler.step()

def change_permissions_recursive(path, mode=0o777):
        for root, dirs, files in os.walk(path, topdown=False):
            for dir in [os.path.join(root,d) for d in dirs]:
                os.chmod(dir, mode)
        for file in [os.path.join(root, f) for f in files]:
                os.chmod(file, mode)

def main(args):
    utils.init_distributed_mode(args)

    # location of main folder at local server
    local_dir = '/home/user/jusung/weight_direction/'

    # save location of FTP server and location of local sub-folder
    copy_dir = '{}/{}/num_data_{}/batch_{}_pretrain_backbone_{}_aux_{}/WD_{}_lr_{}_{}_decay_{}_warmup_{}_moment_{}_epoch{}_'.format(
        args.output_dir, args.model, args.num_sample, args.batch_size, args.pretrained_backbone, args.aux_loss, \
                                                        args.weight_decay, args.lr, args.schedule_type, \
                                                        str(args.decay_epoch).replace("[", "").replace("]", "").replace(",", "_"), \
                                                        args.lr_warmup_epochs, args.momentum, args.epochs)

    i=0
    while os.path.isdir(copy_dir + str(i)) or os.path.isdir(local_dir+copy_dir + str(i)):   # whether to exist in FTP or local 
        i += 1
    copy_dir = copy_dir + str(i)

    args.output_dir = local_dir+copy_dir
    if utils.is_main_process():
        os.makedirs(args.output_dir, exist_ok=True)
        change_permissions_recursive(local_dir, 0o777)
        # os.chmod(args.output_dir, 0o777)
        # configure(args.output_dir)
        print(args.output_dir)

    # print(args)
    if (not args.distributed) or (args.distributed and utils.is_main_process()):
        logger = logging.getLogger("js_logger")
        fileHandler = logging.FileHandler(args.output_dir+'/train.log')
        streamHandler = logging.StreamHandler()
        logger.addHandler(fileHandler)
        logger.addHandler(streamHandler)
        logger.setLevel(logging.INFO)
        logger.info(args)

        if args.tensorboard: configure(args.output_dir)
    else:
        logger = None

    device = torch.device(args.device)

    dataset, num_classes = get_dataset(args.data_path, args.dataset, "train", get_transform(train=True), num_sample=args.num_sample)
    dataset_test, _ = get_dataset(args.data_path, args.dataset, "val", get_transform(train=False))
    print(len(dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn, drop_last=True)        # collate는 뭐하는건지???

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    model = torchvision.models.segmentation.__dict__[args.model](num_classes=num_classes,               # args.model = fcn_resnet50
                                                                 aux_loss=args.aux_loss,
                                                                 pretrained=args.pretrained,
                                                                 pretrained_backbone = args.pretrained_backbone)
    # print(model)
    model.to(device)
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    else:
        model = torch.nn.DataParallel(model)
        model_without_ddp = model.module

    params_to_optimize = [
        {"params": [p for p in model_without_ddp.backbone.parameters() if p.requires_grad]},
        {"params": [p for p in model_without_ddp.classifier.parameters() if p.requires_grad]},
    ]
    if args.aux_loss:
        params = [p for p in model_without_ddp.aux_classifier.parameters() if p.requires_grad]
        params_to_optimize.append({"params": params, "lr": args.lr * 10})
    optimizer = torch.optim.SGD(
        params_to_optimize,
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    iters_per_epoch = len(data_loader)
    if args.schedule_type == 'exp':
        main_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (iters_per_epoch * (args.epochs - args.lr_warmup_epochs))) ** 0.9)
    elif args.schedule_type == 'step':
        main_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_epoch, gamma=args.lr_decay)
    else:
        raise RuntimeError("Invalid lr scheduling method '{}'. Only exp and step are supported.".format(args.schedule_type))
        
    if args.lr_warmup_epochs > 0:
        warmup_iters = iters_per_epoch * args.lr_warmup_epochs
        args.lr_warmup_method = args.lr_warmup_method.lower()
        if args.lr_warmup_method == 'linear':
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr_warmup_decay,
                                                                    total_iters=warmup_iters)
        elif args.lr_warmup_method == 'constant':
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=args.lr_warmup_decay,
                                                                      total_iters=warmup_iters)
        else:
            raise RuntimeError("Invalid warmup lr method '{}'. Only linear and constant "
                               "are supported.".format(args.lr_warmup_method))
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[warmup_iters]
        )
    else:
        lr_scheduler = main_lr_scheduler

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=not args.test_only)
        if not args.test_only:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.test_only:
        confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, logger=logger)
        if (not args.distributed) or (args.distributed and utils.is_main_process()):
            logger.info(confmat)
        return

    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)      # 이유는 몰루
        train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, args.print_freq, logger, args.schedule_type)
        if (epoch+1) % args.test_freq == 0:
            confmat = evaluate(model, data_loader_test, device=device, num_classes=num_classes, logger=logger)
            if (not args.distributed) or (args.distributed and utils.is_main_process()):
                logger.info(confmat)
        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'args': args
        }
        # utils.save_on_master(
        #     checkpoint,
        #     os.path.join(args.output_dir, 'model_{}.pth'.format(epoch)))
    utils.save_on_master(
        checkpoint,
        os.path.join(args.output_dir, 'checkpoint.pth'))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if utils.is_main_process():
        try: 
            shutil.copytree(args.output_dir, copy_dir)
            os.chmod(copy_dir, 0o777)
        except:
            shutil.copytree(args.output_dir, copy_dir+'_copy2')
            os.chmod(copy_dir+'_copy2', 0o777)


def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch Segmentation Training', add_help=add_help)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset path')
    parser.add_argument('--dataset', default='coco', help='dataset name')
    parser.add_argument('--model', default='fcn_resnet101', help='model')
    parser.add_argument('--aux_loss', action='store_true', help='auxiliar loss')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch_size', default=8, type=int)
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.01, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--schedule_type', default='step', type=str, help='learning rate scheduling type')                            
    parser.add_argument('--decay_epoch', type=int, nargs='+', default=[45,55], help='Learning Rate Decay Steps')
    parser.add_argument('--lr_decay', default=0.1, type=float, help='learning rate decay ratio')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='the number of epochs to warmup (default: 0)')
    parser.add_argument('--lr-warmup-method', default="linear", type=str, help='the warmup method (default: linear)')
    parser.add_argument('--lr-warmup-decay', default=0.01, type=float, help='the decay for lr')
    parser.add_argument('--print_freq', default=10, type=int, help='print frequency')
    parser.add_argument('--test_freq', default=1, type=int, help='test frequency')
    parser.add_argument('--output-dir', default='./logs_coco', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--test-only",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained_backbone",
        dest="pretrained_backbone",
        help="Use ImageNet pre-trained models from the modelzoo",
        action="store_true",
    )
    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # JS
    parser.add_argument('--num_sample', default=80000, type=int, help='number of training samples')   
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    
    main(args)
