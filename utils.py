# Built-in Imports
import os
from typing import Callable, Optional, Tuple, List, Dict, Any
from io import BytesIO
from zipfile import ZipFile
import argparse

# Image Processing and CV Imports
import cv2
import numpy as np

# PyTorch Imports
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T
from vision.references.detection import transforms as T_



class DroneDataset(Dataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super(DroneDataset, self).__init__()

        fd = open(root, 'rb')
        zip_content = fd.read()
        fd.close()
        self.zip_file = ZipFile( BytesIO(zip_content), 'r' )

        img_prefix = 'data/images/'
        img_list = (
            list(filter( 
                lambda x: x[:len(img_prefix)] == img_prefix and x[-4:] == '.jpg',
                self.zip_file.namelist() 
            ))
        )

        ann_prefix = 'data/annotations/'
        ann_list = (
            list(filter( 
                lambda x: x[:len(ann_prefix)] == ann_prefix and x[-4:] == '.txt',
                self.zip_file.namelist() 
            ))
        )

        for img, ann in zip(img_list, ann_list):
            if img[len(img_prefix):-4] != ann[len(ann_prefix):-4]:
                raise RuntimeError('Images and Annotations are not sorted in the correct order')
        
        self.img_list = img_list
        self.ann_list = ann_list

        self.transforms = transforms
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Resize((320, 320)),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    

    def __getitem__(self, idx: int) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        img_path = self.img_list[idx]
        ann_path = self.ann_list[idx]

        img_buf = self.zip_file.read(name=img_path)
        img = cv2.imdecode(np.frombuffer(img_buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = self.preprocess( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )

        targets = {}
        boxes = []
        labels = []
        scores = []
        with self.zip_file.open(ann_path) as fd:
            for line in fd:
                line = line.decode(encoding='utf-8').split(',')
                x1, y1, w, h, score, label, _, _ = line
                x1, y1, w, h, score = int(x1), int(y1), int(w), int(h), int(score)
                x2 = x1 + w
                y2 = y1 + h
                boxes.append( [x1, y1, x2, y2] )
                labels.append(label)
                scores.append(score)
        
        targets['boxes'] = torch.as_tensor(boxes)
        targets['labels'] = torch.as_tensor(labels)
        targets['scores'] = torch.as_tensor(scores)

        if self.transforms is not None:
            img, targets = self.transforms(img, targets)
        
        return img, targets
    

    def __len__(self) -> int:
        return len(self.img_list)



def parse():
    parser = argparse.ArgumentParser()

    # Training parameters
    parser.add_argument('--train', action='store_true', help='Run training script')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for training')
    parser.add_argument('--optim', type=str, default='adam', help='optimizer used during training')
    parser.add_argument('--scheduler', type=str, default='step', help='Scheduler for adaptive learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay to prevent weight explosion')
    parser.add_argument('--step_size', type=int, default=3, help='step size for step lr scheduler')
    parser.add_argument('--gamma', type=float, default=0.975, help='Decay for step LR scheduler')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of Epochs to train')
    parser.add_argument('--log_interval', type=int, default=5, help='Frequency of logging checkpoints')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training and validation')
    parser.add_argument('--num_batches', type=int, default=330, help='Total training batches for training and validation split as 90/10')
    parser.add_argument('--aug', action='store_true', help='Flag to enable augmentation with Gaussian noise')

    # General parameters
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default='data/', help='Root dir of data')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
    parser.add_argument('--visualize', action='store_true', help='flag to visualize some results')

    # Evaluate Existing Model
    parser.add_argument('--eval_pth', action='store_true', help='Evaluate model from a .pth file')
    parser.add_argument('--eval_ckpt', action='store_true', help='Evaluate model from checkpoint')

    # Model Config
    parser.add_argument('--use_pretrained', action='store_true', help='Uses pretrained Imagenet weights for MobileNetv3 backbone')

    args = parser.parse_args()
    return args


def get_dataset(root: str, transforms: Optional[Callable] = None) -> Dataset:
    if not os.path.exists(root):
        raise ValueError(f'Data root: {root} does not exist')
    dataset = DroneDataset(root, transforms)
    return dataset


def get_loaders(args: Any) -> Tuple[DataLoader, DataLoader]:
    augment = None
    if args.aug:
        augment = T_.Compose([
            T_.RandomHorizontalFlip(0.5)
        ])
    dataset = get_dataset(args.data_dir, augment)
    
    val_batches = (args.num_batches // 11) * args.batch_size
    train_batches = (args.num_batches - (args.num_batches // 11) ) * args.batch_size    
    train_data, val_data, _ = torch.utils.data.random_split(
        dataset, [train_batches, val_batches, len(dataset) - args.batch_size*args.num_batches]
    )

    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, args.batch_size, num_workers=2)

    return train_loader, val_loader