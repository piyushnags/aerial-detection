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
    
    # Ge trainable params and move the model to device
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("No. of trainable parameters: {}".format(trainable))
    model.to(device)

    # Set optimizer
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
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
        evaluate(model, val_loader, device)

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
    
    elif args.eval_ckpt or args.eval_pth:
        raise NotImplementedError()