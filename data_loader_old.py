'''
SVHN is not completely implemented 
'''

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import os


def getIMAGENET(batch_size, data_root='/home/user/ssd1/data/ILSVRC2012/', train=True, val=True, **kwargs):
    ds = []
    traindir = os.path.join(data_root, 'train')
    valdir = os.path.join(data_root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    num_workers = kwargs.setdefault('num_workers', 32)
    drop_last = kwargs.setdefault('drop_last', True)
    # if args.distributed:
    #     train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    # else:
    # train_sampler = None

    if train:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=drop_last)
        ds.append(train_loader)

    if val:
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        ds.append(val_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


class index_CIFAR10(Dataset):
    # to index previous training dataset in active_js_1_4.py
    # it is used for distilling previous network 
    def __init__(self, root, train, download, transform):
        self.cifar10 = datasets.CIFAR10(root=root, train=train, download=download, transform=transform)
        
    def __getitem__(self, index):
        data, target = self.cifar10[index]
        
        # Your transformations here (or set it in CIFAR10)
        
        return data, target, index

    def __len__(self):
        return len(self.cifar10)

class index_CIFAR100(Dataset):
    # to index previous training dataset in active_js_1_4.py
    def __init__(self, root, train, download, transform):
        self.cifar100 = datasets.CIFAR100(root=root, train=train, download=download, transform=transform)
        
    def __getitem__(self, index):
        data, target = self.cifar100[index]
        
        # Your transformations here (or set it in CIFAR10)
        
        return data, target, index

    def __len__(self):
        return len(self.cifar100)

class index_MNIST(Dataset):
    # to index previous training dataset in active_js_1_4.py
    def __init__(self, root, train, download, transform):
        self.mnist = datasets.MNIST(root=root, train=train, download=download, transform=transform)
        
    def __getitem__(self, index):
        data, target = self.mnist[index]
        
        # Your transformations here (or set it in CIFAR10)
        
        return data, target, index

    def __len__(self):
        return len(self.mnist)


def getCIFAR10(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    drop_last = kwargs.setdefault('drop_last', True)
    shuffle = kwargs.setdefault('shuffle', True)
    sampler = kwargs.setdefault('sampler', None)
    index = kwargs.pop('index', False)

    if 'transform_train' in kwargs:
        transform_train = kwargs.pop('transform_train')
    else:
        transform_train=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])   
    if 'transform_test' in kwargs:
        transform_test = kwargs.pop('transform_test')
    else:
        transform_test=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])             
    # print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        if index:
            dataset = index_CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        else:
            dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
        if sampler is not None:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transform_test),
            batch_size=batch_size, pin_memory=True, shuffle=False, num_workers=num_workers, drop_last=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getCIFAR100(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar100-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    drop_last = kwargs.setdefault('drop_last', True)
    shuffle = kwargs.setdefault('shuffle', True)
    sampler = kwargs.setdefault('sampler', None)
    index = kwargs.pop('index', False)

    if 'transform_train' in kwargs:
        transform_train = kwargs.pop('transform_train')
    else:
        transform_train=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])
    if 'transform_test' in kwargs:
        transform_test = kwargs.pop('transform_test')
    else:
        transform_test=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
            ])        
    # print("Building CIFAR-100 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        if index:
            dataset = index_CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        else:
            dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform_train)
        if sampler is not None:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root=data_root, train=False, download=True,
                transform=transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getMNIST(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, ',mnist-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    drop_last = kwargs.setdefault('drop_last', True)
    shuffle = kwargs.setdefault('shuffle', True)
    sampler = kwargs.setdefault('sampler', None)
    index = kwargs.pop('index', False)

    transform_train = transforms.Compose([
                                     transforms.ToTensor(), # image to Tensor
                                     transforms.Normalize((0.1307,), (0.3081,)) # image, label
                                 ])
    transform_test = transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307, ),(0.3081, ))
                               ])
    ds = []
    if train:
        if index:
            dataset = index_MNIST(root=data_root, train=True, download=True, transform=transform_train)
        else:
            dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform_train)

        if sampler is not None:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, \
                                sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, \
                                shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                root=data_root, train=False, download=True,
                transform=transform_test),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def getSVHN(batch_size, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 2)
    drop_last = kwargs.setdefault('drop_last', True)
    shuffle = kwargs.setdefault('shuffle', True)
    sampler = kwargs.setdefault('sampler', None)
    # index = kwargs.pop('index', False)
    # kwargs.pop('input_size', None)
    # print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    if 'transform_train' in kwargs:
        transform_train = kwargs.pop('transform_train')
    else:
        transform_train=transforms.Compose([
                # transforms.RandomCrop(32),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])
    if 'transform_test' in kwargs:
        transform_test = kwargs.pop('transform_test')
    else:
        transform_test=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
            ])        

    ds = []
    if train:
        dataset = datasets.SVHN(root=data_root, split='train', download=True, transform=transform_train, target_transform=target_transform)
        if sampler is not None:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                sampler=sampler, num_workers=num_workers, drop_last=drop_last)
        else:
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=True, \
                                shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=transform_test,
                target_transform=target_transform
            ),
            batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds


def getDataSet(data_type, batch_size, dataroot, **kwargs):
    if data_type == 'cifar10':  
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, data_root=dataroot, **kwargs)
    elif data_type == 'cifar100':  
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, data_root=dataroot, **kwargs)
    elif data_type == 'svhn':  
        train_loader, test_loader = getSVHN(batch_size=batch_size, data_root=dataroot, num_workers=1, **kwargs)    
    elif data_type =='mnist':
        train_loader, test_loader = getMNIST(batch_size=batch_size, data_root=dataroot, **kwargs)    
    elif data_type == 'indexed_cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, data_root=dataroot, index=True, **kwargs)
    elif data_type == 'indexed_cifar100':
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, data_root=dataroot, index=True, **kwargs)
    elif data_type == 'indexed_mnist':
        train_loader, test_loader = getMNIST(batch_size=batch_size, data_root=dataroot, index=True, **kwargs)
    # for calculating train set entropy (active learning)
    elif data_type == 'cifar10_eval':
        transform_train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, data_root=dataroot, transform_train=transform_train, shuffle=False, **kwargs)
    # for calculating train set entropy (active learning) 
    elif data_type == 'cifar100_eval':
        transform_train=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_loader, test_loader = getCIFAR100(batch_size=batch_size, data_root=dataroot, transform_train=transform_train, shuffle=False, **kwargs)
    # for calculating train set entropy (active learning) 
    elif data_type == 'mnist_eval':
        train_loader, test_loader = getMNIST(batch_size=batch_size, data_root=dataroot, shuffle=False, **kwargs)

    # old... do not use
    elif data_type == 'cifar10_augment_eval':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, data_root=dataroot, shuffle=False, **kwargs)   

    elif data_type == 'imagenet':
        train_loader, test_loader = getIMAGENET(batch_size=batch_size, data_root=dataroot, **kwargs)           
    return train_loader, test_loader