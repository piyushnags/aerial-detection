import torch
import torch.nn as nn
import torchvision
from models import *
from utils import *
from tqdm import tqdm
from torch.utils.data import DataLoader


def train_one_epoch(model, train_loader, device, optimizer, epoch, freq):
    model.train()
    pass


def evaluate(model, val_loader, device):
    model.eval()
    pass


def train():
    pass