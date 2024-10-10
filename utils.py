import os
import math
import PIL
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

import datasets

def adjust_lr(optimizer, lr):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group['lr'] = lr
        if i > 0:#Linear layers for CLIP ViTs
            param_group['lr'] *= 10000

def cosine_lr(init_lr, stage1, epochs):
    lrs = [init_lr] * epochs
    init_lr_stage_ldl = init_lr
    for t in range(stage1, epochs):
        lrs[t] = 0.5 * init_lr_stage_ldl * (1 + math.cos((t - stage1 + 1) * math.pi / (epochs - stage1 + 1)))
    return lrs

def min_max(x):
    return (x - x.min())/(x.max() - x.min())

def multi_class_loss(pred, target):
    pred = F.softmax(pred, dim=1)
    loss = - torch.sum(target*torch.log(pred), dim=1)
    return loss

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1-lam)
    else:
        lam = 1

    device = x.get_device()
    batch_size = x.size()[0]
    
    index = torch.randperm(batch_size)        
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam, index

def get_dataset_hyper(dataset):
    if dataset == 'miniimagenet':
        mean = [0.4728, 0.4487, 0.4031]
        std = [0.2744, 0.2663 , 0.2806]
        size1 = 32
        size = 32
    elif dataset == 'webvision':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size1 = 256
        size = 224
    elif dataset == 'food101n':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size1 = 256
        size = 224
    elif 'web-' in dataset:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        size1 = 512
        size = 448
    elif dataset == 'custom':
        #Replace by the values for your dataset
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        size1 = 256
        size = 224   

    return mean, std, size1, size
    

def make_data_loader(args, no_aug=False, transform=None, **kwargs):

    mean, std, size1, size = get_dataset_hyper(args.dataset)
    if "clip" in args.net:
        #Default CLIP image size values
        size1 = 256
        size = 224

    
    transform_train = transforms.Compose([        
        transforms.Resize(size1, interpolation=PIL.Image.BICUBIC, antialias=True),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    
    transform_track = transforms.Compose([
        transforms.Resize(size1, interpolation=PIL.Image.BICUBIC, antialias=True),
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
    ])    

    transform_test = transforms.Compose([
        transforms.Resize(size1, interpolation=PIL.Image.BICUBIC, antialias=True),
        transforms.CenterCrop(size),
    ])
        
    transform_cont = transforms.Compose([
        transforms.Resize(size1, interpolation=PIL.Image.BICUBIC, antialias=True),
        transforms.RandomResizedCrop(size, interpolation=PIL.Image.BICUBIC, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    rand1, rand2 = 1, 6
    if args.dataset not in ['miniimagenet']:
        rand1, rand2 = 1, 4
        
    transform_ssl = transforms.Compose([
        transforms.Resize(size1, interpolation=PIL.Image.BICUBIC, antialias=True),
        transforms.RandomResizedCrop(size, interpolation=PIL.Image.BICUBIC, antialias=True),
        transforms.RandAugment(rand1,rand2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    clean_noisy = None
    if args.dataset == "miniimagenet":
        from datasets.miniimagenet import make_dataset, MiniImagenet
        train_data, train_labels, val_data, val_labels, test_data, test_labels, clean_noisy = make_dataset(noise_ratio=args.noise_ratio, seed=args.seed)
        trainset = MiniImagenet(train_data, train_labels, transform=transform_train, transform_cont=transform_cont, transform_ssl=transform_ssl, clean_noisy=clean_noisy)
        trackset = MiniImagenet(train_data, train_labels, transform=transform_test, transform_cont=None, clean_noisy=clean_noisy)
        testset = MiniImagenet(val_data, val_labels, transform=transform_test)
    elif "web-" in args.dataset:
        from datasets.web_fg import fg_web_dataset
        trainset = fg_web_dataset(transform=transform_train, mode="train", transform_cont=transform_cont, transform_ssl=transform_ssl, which=args.dataset)
        trackset = fg_web_dataset(transform=transform_test, mode="train", which=args.dataset)
        testset = fg_web_dataset(transform=transform_test, mode="test", which=args.dataset)
        clean_noisy = torch.ones(len(trainset), dtype=torch.bool)
    elif args.dataset == "webvision":
        from datasets.webvision import webvision_dataset
        trainset = webvision_dataset(mode="train", num_classes=50, transform=transform_train, transform_cont=transform_cont, transform_ssl=transform_ssl)
        trackset = webvision_dataset(mode="train", num_classes=50, transform=transform_test, transform_cont=None)
        testset = webvision_dataset(mode="test", num_classes=50, transform=transform_test)
        clean_noisy = torch.ones(len(trainset), dtype=torch.bool)
    elif args.dataset == "custom":
        from datasets.custom import make_dataset, Custom
        train_data, train_labels, val_data, val_labels, test_data, test_labels = make_dataset()
        trainset = Custom(train_data, train_labels, transform=transform_train, transform_cont=transform_cont, transform_ssl=transform_ssl)
        trackset = Custom(train_data, train_labels, transform=transform_test)
        testset = Custom(val_data, val_labels, transform=transform_test)        
    else:
        raise NotImplementedError("Dataset {} is not implemented".format(args.dataset))
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs) #Normal training        
    track_loader = torch.utils.data.DataLoader(trackset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    return train_loader, test_loader, track_loader, clean_noisy

def create_save_folder(args):
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
    if not os.path.isdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset)):
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset))
    if not os.path.isdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name)):
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name))
    if not os.path.isdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name, str(args.seed))):
        os.mkdir(os.path.join(args.save_dir, args.net + '_'  + args.dataset, args.exp_name, str(args.seed)))
