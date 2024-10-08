import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from autoaugment import CIFAR10Policy, SVHNPolicy, ImageNetPolicy
from criterions import LabelSmoothingCrossEntropyLoss
from da import RandomCropPaste


def get_criterion(args):
    if args.criterion=="ce":
        if args.label_smoothing:
            criterion = LabelSmoothingCrossEntropyLoss(args.num_classes, smoothing=args.smoothing)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"{args.criterion}?")

    return criterion

def get_model(args):
    
    if args.vit_type == 'vit_tiny':     # 5.8M
        args.num_layers=12
        args.hidden=192
        args.mlp_hidden=768
        args.head=3
        
    elif args.vit_type == 'vit_small':    # 22.2M
        args.num_layers=12
        args.hidden=384
        args.mlp_hidden=1536
        args.head=6
        
    elif args.vit_type == 'vit_base':    # 86M
        args.num_layers=12
        args.hidden=768
        args.mlp_hidden=3072
        args.head=12
        
    else:
        pass 
        
        
    if args.model_name == 'vit_splithead':
        from vit_splithead import ViT_SplitHead
        
        net = ViT_SplitHead(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token,
            head_mix_method=args.head_mix_method 
            )

        print('vit splithead loaded')
        
        
    elif args.model_name == 'vit_orig':
        from vit_orig import ViT_Orig
              
        net = ViT_Orig(
            args.in_c, 
            args.num_classes, 
            img_size=args.size, 
            patch=args.patch, 
            dropout=args.dropout, 
            mlp_hidden=args.mlp_hidden,
            num_layers=args.num_layers,
            hidden=args.hidden,
            head=args.head,
            is_cls_token=args.is_cls_token
            )
        
        print('vit orig loaded')
   
    args.tot_param = sum(i.numel() for i in net.parameters()) / 1e6
    print('############################## RUN INFO ##############################')
    print('WANDB EXP NAME: ', args.experiment_name)
    print('MODEL NAME', args.model_name)
    print('VIT TYPE: ', args.vit_type)
    print('HEAD MIX: ', args.head_mix_method)
    print(f'TOTAL PARAMS: {args.tot_param:.3f}M')
    print('############################## RUN INFO ##############################')

    breakpoint()
    
    return net

def get_transform(args):
    train_transform = []
    test_transform = []
    
    if args.dataset == 'c10' or args.dataset == 'c100':
        train_transform += [transforms.RandomCrop(size=args.size, padding=args.padding),
                            transforms.RandomHorizontalFlip()]
        if args.autoaugment:
            train_transform += [CIFAR10Policy()]
        train_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)]
        if args.rcpaste:
            train_transform += [RandomCropPaste(size=args.size)]
            
        test_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)]
    
    elif args.dataset == 'svhn':
        train_transform += [transforms.RandomCrop(size=args.size, padding=args.padding),]
        if args.autoaugment:
            train_transform += [SVHNPolicy()]
        train_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)]
        if args.rcpaste:
            train_transform += [RandomCropPaste(size=args.size)]
            
        test_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)]
        
    elif args.dataset == 'imagenet':
        train_transform += [
            transforms.RandomResizedCrop(args.size),
            transforms.RandomHorizontalFlip(),]
        if args.autoaugment:
            train_transform.append(ImageNetPolicy())
        train_transform += [
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)]
        
        test_transform += [
            transforms.Resize(256),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=args.mean, std=args.std)]
    
    train_transform = transforms.Compose(train_transform)
    test_transform = transforms.Compose(test_transform)

    return train_transform, test_transform
    

def get_dataset(args):
    root = args.dataset_path
    if args.dataset == "c10":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR10(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR10(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "c100":
        args.in_c = 3
        args.num_classes=100
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.CIFAR100(root, train=True, transform=train_transform, download=True)
        test_ds = torchvision.datasets.CIFAR100(root, train=False, transform=test_transform, download=True)

    elif args.dataset == "svhn":
        args.in_c = 3
        args.num_classes=10
        args.size = 32
        args.padding = 4
        args.mean, args.std = [0.4377, 0.4438, 0.4728], [0.1980, 0.2010, 0.1970]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.SVHN(root, split="train",transform=train_transform, download=True)
        test_ds = torchvision.datasets.SVHN(root, split="test", transform=test_transform, download=True)
    
    elif args.dataset == 'imagenet':
        args.in_c = 3
        args.num_classes=1000
        args.size = 224
        args.mean, args.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        train_transform, test_transform = get_transform(args)
        train_ds = torchvision.datasets.ImageNet(root, split="train",transform=train_transform)
        test_ds = torchvision.datasets.ImageNet(root, split="val", transform=test_transform)
        

    else:
        raise NotImplementedError(f"{args.dataset} is not implemented yet.")
    
    return train_ds, test_ds

def get_experiment_name(args):
    if args.model_name == 'vit_splithead':
        experiment_name = f"server8_gpu{args.gpu}_{args.dataset}_{args.model_name}_{args.vit_type}_numpatch{args.patch}_method{args.head_mix_method}_batch:{args.batch_size}_lr:{args.lr}_wd:{args.weight_decay}_warm:{args.warmup_epoch}_drop:{args.dropout}"
    elif args.model_name == 'vit_orig':
        experiment_name = f"server8_gpu{args.gpu}_{args.dataset}_{args.model_name}_{args.vit_type}_numpatch{args.patch}_batch:{args.batch_size}_lr:{args.lr}_wd:{args.weight_decay}_warm:{args.warmup_epoch}_drop:{args.dropout}"
    
    # experiment_name = f"tes"
    if args.autoaugment:
        experiment_name+="_aa"
    if args.label_smoothing:
        experiment_name+="_ls"
    if args.rcpaste:
        experiment_name+="_rc"
    if args.cutmix:
        experiment_name+="_cm"
    if args.mixup:
        experiment_name+="_mu"
    if args.off_cls_token:
        experiment_name+="_gap"
    print(f"Experiment:{experiment_name}")
    return experiment_name
