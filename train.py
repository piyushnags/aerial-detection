# Built-in Imports
import math, sys

# PyTorch Imports 
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam, SGD, RMSprop
import torchvision

# Project Imports
from models import *
from utils import *
from tqdm import tqdm


def train_one_epoch(model, train_loader, device, optimizer, epoch, freq):
    model.train()

    batch_loss = []
    
    for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        batch_loss.append(loss_value)
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if batch_idx % freq == 0:
            print(f'Epoch {epoch} Batch {batch_idx+1}, Loss: {loss_value}')
    
    avg_loss = sum(batch_loss)/len(batch_loss)
    print(f'Epoch {epoch} Average Loss: {avg_loss}')
    return avg_loss


# FIXME: Update function to handle size mismatch or
# use a split of training data to validate results
def evaluate(model, val_loader, device):
    model.eval()

    batch_loss = []
    misclf = 0

    for images, targets in tqdm(val_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images)        

        losses = [
            torch.mean([ 
                torch.abs(output['boxes'] - t['boxes']) 
                for output, t in zip(outputs, targets)
            ]) 
        ]

        loss_value = sum(losses).item()
        batch_loss.append(loss_value)

        for output, t in zip(outputs, targets):
            tmp = output['labels'] - t['labels']
            tmp = torch.where(t != 0, 1, 0.)
            misclf += torch.sum(tmp).item()
        
    misclf /= len(val_loader)
    avg_loss = sum(batch_loss)/len(batch_loss)
    print(f'Average Evaluation Loss: {avg_loss}, Avg. Misclfs per Batch: {misclf}')
    return avg_loss, misclf


def train(args: Any, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader):
    print(f'Number of training samples: {args.batch_size*len(train_loader)} samples')
    print(f'Number of validation samples: {args.batch_size*len(val_loader)} samples')

    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    epochs = args.num_epochs
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print("No. of trainable parameters: {}".format(trainable))
    model.to(device)

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Optimizer {args.optim} not supported currently')

    scheduler = None
    if args.scheduler == 'step':
        scheduler = StepLR(optimizer, args.step_size, args.gamma)
    
    train_losses = []
    val_losses = []
    misclfs = []

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    for epoch in range(1, epochs+1):
        l1 = train_one_epoch(model, train_loader, device, optimizer, epoch, args.print_freq)
        
        # FIXME: Enable corrected version of evaluate func
        l2, misclf = evaluate(model, val_loader, device)
        # eval(model, val_loader, device)
        
        if scheduler is not None:
            scheduler.step()
        
        train_losses.append(l1)
        val_losses.append(l2)
        misclfs.append(misclf)

        # FIXME: Uncomment saving val losses and misclfs
        if epoch % args.log_interval == 0:
            torch.save(
                {
                    "epoch":epoch,
                    "model_state_dict":model.state_dict(),
                    "optimizer_state_dict":optimizer.state_dict(),
                    "training_losses":train_losses,
                    "val_losses":val_losses,
                    "misclfs":misclfs,
                    "scheduler_state_dict":scheduler.state_dict()
                },
                os.path.join(args.save_dir, f'ckpt_{epoch}.ckpt')
            )

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'model.pth'))
    
    # FIXME: Return all losses and misclassifications
    return train_losses, val_losses, misclfs
    # return train_losses



if __name__ == '__main__':
    args = parse()
    if args.train:
        train_loader, val_loader = get_loaders(args)
        model = SSDLite(num_classes=args.num_classes, pretrained=args.use_pretrained)
        train_losses, val_losses, misclfs = train(args, model, train_loader, val_loader)
        # FIXME: temporary hack
        # train_losses = train(args, model, train_loader, val_loader)
        # val_losses, misclfs = [], []
        plot_stats(args, train_losses, val_losses, misclfs)
    
    elif args.eval_ckpt or args.eval_pth:
        raise NotImplementedError()