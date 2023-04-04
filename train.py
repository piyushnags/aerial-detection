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


def train_one_epoch(model, train_loader, device, optimizer, epoch, freq, args):
    model.train()

    batch_loss = []

    lr_scheduler = None
    if args.scheduler == 'step':
        lr_scheduler = StepLR(optimizer, args.step_size, args.gamma)
    
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

        if lr_scheduler is not None:
            lr_scheduler.step()
        
        if batch_idx % freq == 0:
            print(f'Epoch {epoch} Batch {batch_idx+1}, Loss: {loss_value}')
    
    avg_loss = sum(batch_loss)/len(batch_loss)
    print(f'Epoch {epoch} Average Loss: {avg_loss}')
    return avg_loss


def evaluate(model, val_loader, device):
    model.eval()

    batch_loss = []
    misclf = 0

    for images, targets in tqdm(val_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)

        losses = [
            torch.mean( 
                torch.abs(output['boxes'] - t['boxes']) 
                for output, t in zip(outputs, targets)
            ) 
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


def train():
    pass