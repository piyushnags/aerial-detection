# Built-in Imports
import math, sys

# PyTorch Imports 
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torchvision

# Project Imports
from models import *
from utils import *
from engine import train_one_epoch, evaluate



def train(args: Any, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
    # Print basic stats dyamically to catch any obvious errors
    print(f'Number of training samples: {args.batch_size*len(train_loader)} samples')
    print(f'Number of validation samples: {args.batch_size*len(val_loader)} samples')

    # Set device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    epochs = args.num_epochs
    
    # Get trainable params and move the model to device
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("No. of trainable parameters: {}".format(trainable))
    model.to(device)

    # Set optimizer
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    else:
        raise ValueError(f'Optimizer {args.optim} not supported currently')

    # Set scheduler
    scheduler = None
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, args.step_size, args.gamma)

    # Create a dir to save results if it doesn't exist yet
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # Training loop
    for epoch in range(1, epochs+1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=args.print_freq)
        scheduler.step()
        # evaluate(model, val_loader, device)

        # Save checkpoints
        if epoch % args.log_interval == 0:
            torch.save(
                {
                    "epoch":epoch,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "scheduler_state_dict":scheduler.state_dict()
                },
                os.path.join(args.save_dir, f'ckpt_{epoch}.ckpt')
            )

    # Save .pth file of model once training is completed successfully
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))


def resume_training(args: Any):
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"{args.model_path} is an invalid file path to model")
    
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    if args.model_path[-5:] != '.ckpt':
        raise ValueError(f"{args.model_path} is not a valid model checkpoint")
    
    ckpt = torch.load( args.model_path, map_location=device )
    model_state_dict = ckpt['model_state_dict']
    optimizer_state_dict = ckpt['optimizer_state_dict']
    scheduler_state_dict = ckpt['scheduler_state_dict']
    e = ckpt['epoch']
    
    epochs = args.num_epochs

    model = SSDLite(args.num_classes, pretrained=args.use_pretrained)
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("No. of trainable parameters: {}".format(trainable))
    model.to(device)
    model.load_state_dict(model_state_dict)

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        optimizer.load_state_dict( optimizer_state_dict )
    elif args.optim == 'sgd':
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
        optimizer.load_state_dict( optimizer_state_dict )
    else:
        raise ValueError(f"{args.optim} is not a supported optimizer. Please use Adam or SGD instead")
    
    scheduler = None
    if args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        scheduler.load_state_dict( scheduler_state_dict )
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    train_loader, val_loader = get_loaders(args)

    for epoch in range(1, epochs+1):
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=args.print_freq)
        scheduler.step()
        # evaluate(model, val_loader, device)

        # Save checkpoints
        if epoch % args.log_interval == 0:
            torch.save(
                {
                    "epoch":epoch,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "scheduler_state_dict":scheduler.state_dict()
                },
                os.path.join(args.save_dir, f'ckpt_{epoch+e}.ckpt')
            )
        
    # Save .pth file of model once training is completed successfully
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))    
        


if __name__ == '__main__':
    torch.manual_seed(2023)
    args = parse()
    if args.train:
        # Get data loaders
        train_loader, val_loader = get_loaders(args)

        # Initialize model
        model = SSDLite(num_classes=args.num_classes, pretrained=args.use_pretrained)

        # Call training function
        train(args, model, train_loader, val_loader)
    
    elif args.resume:
        resume_training(args)