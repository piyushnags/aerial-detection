# Built-in Imports
import os, sys
from typing import Callable, Optional, Tuple, List, Dict, Any
from io import BytesIO
from zipfile import ZipFile
import argparse

# Image Processing and CV Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# PyTorch Imports
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T
import transforms as T_
from models import SSDLite



class AddNoise():
    def __init__(self, var=0.1, mean=0.):
        self.std = var**0.5
        self.mean = mean
    

    def __call__(self, x: Tensor, targets: List[Dict]) -> Tensor:
        x += torch.randn(x.size())*self.std + self.mean
        return torch.clamp(x, 0, 1), targets



class PennFudanDataset(Dataset):
    def __init__(self, root: str, transforms: Optional[Callable] = None):
        super(PennFudanDataset, self).__init__()

        # Read the contents of zip file into memory
        fd = open(root, 'rb')
        zip_content = fd.read()
        fd.close()
        self.zip_file = ZipFile( BytesIO(zip_content), 'r' )

        # Get filepaths of images from image sub-dir
        img_prefix = 'PennFudanPed/PNGImages/'
        img_list = (
            list(filter( 
                lambda x: x[:len(img_prefix)] == img_prefix and x[-4:] == '.png',
                self.zip_file.namelist() 
            ))
        )

        # Get filepaths of annotations from image sub-dir
        ann_prefix = 'PennFudanPed/Annotation/'
        ann_list = (
            list(filter( 
                lambda x: x[:len(ann_prefix)] == ann_prefix and x[-4:] == '.txt',
                self.zip_file.namelist() 
            ))
        )

        # Sanity check to make sure images and annotations aren't mixed up
        for img, ann in zip(img_list, ann_list):
            if img[len(img_prefix):-4] != ann[len(ann_prefix):-4]:
                raise RuntimeError('Images and Annotations are not sorted in the correct order')
        
        self.img_list = img_list
        self.ann_list = ann_list

        self.transforms = transforms
        
        # SSDLite model takes care of most of the preprocessing
        # so just convert image to tensor
        self.preprocess = T.Compose([
            T.ToTensor(),
        ])


    def __getitem__(self, idx: int) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        img_path = self.img_list[idx]
        ann_path = self.ann_list[idx]

        # Read the image in RGB format
        img_buf = self.zip_file.read(name=img_path)
        img = cv2.imdecode(np.frombuffer(img_buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = self.preprocess( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )

        # Initialize data structures for targets
        targets = {}
        boxes = []
        labels = []
        iscrowd = []
        image_id = []
        area = []

        # Parse the annotations text file
        # Comments begin with '#', so ignore those lines
        # We only need bbox lines 
        with self.zip_file.open(ann_path) as fd:
            for line in fd:
                if line[0] == "#":
                    continue
                else:
                    line = line.decode(encoding='utf-8').split(':')
                    if 'Bounding box' in line[0]:
                        tmp = line[1].split('-')
                        tmp = list(map(lambda x: x.strip(), tmp))
                        tmp = list( map( lambda x: x[1:-1].split(','), tmp ) )
                        xmin, ymin = tmp[0]
                        xmax, ymax = tmp[1]
                        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                        boxes.append([ xmin, ymin, xmax, ymax ])


        # Convert bboxes to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        targets['boxes'] = boxes

        # Only 1 class, so initialize all objects with 1
        # 0 is for background class
        labels = torch.ones( (len(boxes),), dtype=torch.int64 )
        targets['labels'] = labels

        # Ignore crowds, irrelevant for the purpose of this project
        # Initialize it with zeros
        iscrowd = torch.zeros( (len(boxes),), dtype=torch.int64 )
        targets['iscrowd'] = iscrowd

        # Compute the area of all bounding boxes
        # all hail vectorized operations
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targets['area'] = area

        # image ids are just the index, make a 1-D tensor
        image_id = torch.tensor([idx])
        targets['image_id'] = image_id

        # Apply transforms, if any
        if self.transforms is not None:
            img, targets = self.transforms(img, targets)
        
        return img, targets
    

    def __len__(self) -> int:
        return len(self.img_list)



class DroneFaceDataset(Dataset):
    def __init__(self, data_root: str, ann_path: str, transforms: Optional[Callable] = None):
        super(DroneFaceDataset, self).__init__()

        # Check if root path exists
        if not os.path.exists(data_root):
            raise ValueError(f"Path {data_root} does not exist!")
        
        # Check if annotations path is valid
        if not os.path.exists(ann_path):
            raise ValueError(f"Annotation path {ann_path} is invalid!")

        # Read zipped images into memory
        with open(data_root, 'rb') as fd:
            zip_content = fd.read()
        self.zip_file = ZipFile(BytesIO(zip_content), 'r')

        # Get filepaths of images from image sub-dir
        img_prefix = 'droneface/photos_all/'
        img_list = (
            list(filter( 
                lambda x: x[:len(img_prefix)] == img_prefix and x[-4:] == '.JPG',
                self.zip_file.namelist() 
            ))
        )
        self.img_list = img_list

        # Save annotation file path
        self.ann_path = ann_path

        self.transforms = transforms
        self.preprocess = T.Compose([
            T.ToTensor()
        ])


    def __getitem__(self, idx: int) -> Tuple[ List[Tensor], List[Dict[str, Tensor]] ]:
        fname = self.img_list[idx]

        # Read the image in RGB format
        img_buf = self.zip_file.read(name=fname)
        img = cv2.imdecode(np.frombuffer(img_buf, dtype=np.uint8), cv2.IMREAD_COLOR)
        img = self.preprocess( cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )

        targets = {}
        # Filename without full path
        k = fname.split('/')[-1]
        boxes = []

        # max boxes for this image
        num_boxes = None
        c = 0
        with open(self.ann_path, 'r') as fd:
            for i, line in enumerate(fd):
                # Skip headings
                if i == 0:
                    continue

                # matching image title 
                if line[:len(k)] == k:
                    l = line.split(',')
                    x, y, w, h = int(l[6].split(':')[-1]), int(l[7].split(':')[-1]), int(l[8].split(':')[-1]), int(l[9].split(':')[-1][:-2])
                    xmin, ymin, xmax, ymax = x, y, x+w, y+h
                    bbox = [xmin, ymin, xmax, ymax]
                    boxes.append(bbox)
                    
                    # increment region count
                    c += 1

                    if num_boxes is None:
                        num_boxes = int(l[3])

                # break loop once all bboxes are found
                if num_boxes is not None and c == num_boxes:
                    break                    


        # Convert bboxes to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        targets['boxes'] = boxes

        # Only 1 class, so initialize all objects with 1
        # 0 is for background class
        labels = torch.ones( (len(boxes),), dtype=torch.int64 )
        targets['labels'] = labels

        # Ignore crowds, irrelevant for the purpose of this project
        # Initialize it with zeros
        iscrowd = torch.zeros( (len(boxes),), dtype=torch.int64 )
        targets['iscrowd'] = iscrowd

        # Compute the area of all bounding boxes
        # all hail vectorized operations
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targets['area'] = area

        # image ids are just the index, make a 1-D tensor
        image_id = torch.tensor([idx])
        targets['image_id'] = image_id

        # Apply transforms, if any
        if self.transforms is not None:
            img, targets = self.transforms(img, targets)
        
        return img, targets
    

    def __len__(self) -> int:
        return len(self.img_list)
    


class WIDERFaceDataset(Dataset):
    def __init__(self, data_dir: str, annotations: str, transforms: Optional[Callable] = None, split: str = 'train'):
        super(WIDERFaceDataset, self).__init__()

        # Dataset split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"{split} is not a valid split")
        
        # Check if data_dir is valid and save the dir path
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} is not a valid ZipFile containing datset")
        
        data_dir = data_dir.replace('train', split)
        self.data_dir = data_dir

        # Check if annotation path is valid and save the file path
        if not os.path.exists(annotations):
            raise FileNotFoundError(f"{annotations} is not a valid file path")
        
        # Load annotations
        annotations = annotations.replace('train', split)
        self.img_paths, self.offsets = self._load_paths(annotations)
        self.transforms = transforms
        
        self.split = split
        self.ann = annotations
    

    def __getitem__(self, idx) -> Tuple[List[Tensor], List[Dict[str, Tensor]]]:
        img_path, offset = self.img_paths[idx], self.offsets[idx]

        with open(self.ann, 'r') as fd:
            fd.seek(offset)
            fd.readline()
            
            line = fd.readline()
            line = line.strip()
            num_boxes = int(line)

            boxes = []
            for _ in range(num_boxes):
                line = fd.readline()
                line = line.strip()
                line = line.split(" ")
                x, y, w, h = map(float, line[:4])
                xmin, ymin, xmax, ymax = x, y, x+w, y+h
                boxes.append([xmin, ymin, xmax, ymax])
        
        # Get the image as a torch tensor
        prefix = f'WIDER_{self.split}/images/'
        img_path = os.path.join(prefix, img_path)

        with ZipFile(img_path, 'r') as archive:
            with archive.open(img_path) as fd:
                img = Image.open(fd).convert('RGB')
            
        to_tensor = T.ToTensor()
        img = to_tensor(img)

        # Initialize targets dict
        targets = {}

        # Convert bboxes to torch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        targets['boxes'] = boxes

        # Only 1 class, so initialize all objects with 1
        # 0 is for background class
        labels = torch.ones( (len(boxes),), dtype=torch.int64 )
        targets['labels'] = labels

        # Ignore crowds, irrelevant for the purpose of this project
        # Initialize it with zeros
        iscrowd = torch.zeros( (len(boxes),), dtype=torch.int64 )
        targets['iscrowd'] = iscrowd

        # Compute the area of all bounding boxes
        # all hail vectorized operations
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        targets['area'] = area

        # image ids are just the index, make a 1-D tensor
        image_id = torch.tensor([idx])
        targets['image_id'] = image_id

        if self.transforms is not None:
            img, targets = self.transforms(img, targets)

        return img, targets
    

    def _load_paths(self, ann_path: str) -> Tuple[List[str], List[int]]:
        # Read all the lines from ann file
        img_paths = []
        offsets = []
        with open(ann_path, 'r') as fd:
            offset = 0
            for line in fd.readline():
                l = line.strip()
                if l[-4:] != '.jpg':
                    continue
                else:
                    offsets.append(offset)
                    img_paths.append(l)

                offset += len(line)

        return img_paths, offsets        
    

    def __len__(self) -> int:
        return len(self.img_paths)



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
    parser.add_argument('--print_freq', type=int, default=10, help='Frequency for displaying stats (batches)')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size for training and validation')
    parser.add_argument('--num_batches', type=int, default=33, help='Total training batches for training and validation split as 90/10')
    parser.add_argument('--num_workers', type=int, default=2, help='number of worker threads for the dataloaders. Beware of Multiprocessing bugs')
    parser.add_argument('--aug', action='store_true', help='Flag to enable augmentation with Gaussian noise')
    parser.add_argument('--noise_var', type=float, default=0.4, help='Variance of Gaussian noise added during traning (with augmentation)')
    parser.add_argument('--noise_mean', type=float, default=0.3, help='Mean of Gaussian noise added during traning (with augmentation)')

    # General parameters
    parser.add_argument('--save_dir', type=str, default='results')
    parser.add_argument('--data_dir', type=str, default='data/', help='Root dir of data')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on')
    parser.add_argument('--visualize', action='store_true', help='flag to visualize some results')
    parser.add_argument('--ann_path', type=str, default='../drive/MyDrive/Research/annotations.csv', help='path to annotations')

    # Evaluate Existing Model
    parser.add_argument('--eval_pth', action='store_true', help='Evaluate model from a .pth file')
    parser.add_argument('--eval_ckpt', action='store_true', help='Evaluate model from checkpoint')

    # Model Config
    parser.add_argument('--use_pretrained', action='store_true', help='Uses pretrained Imagenet weights for MobileNetv3 backbone')
    parser.add_argument('--num_classes', type=int, default=12, help='Number of classes for the Classification Head')

    # Flags
    parser.add_argument('--fudan', action='store_true', help='Flag to enable training with PennFudan Dataset')
    parser.add_argument('--wider', action='store_true', help='Flag to enable training with WIDER Face Dataset')

    args = parser.parse_args()
    return args


def get_dataset(root: str, ann_path: Optional[str] = '', transforms: Optional[Callable] = None, 
                dset: str = 'fudan', split: Optional[str] = 'train') -> Dataset:    
    # Initialize dataset object and return handle
    
    if dset == 'fudan':
        dataset = PennFudanDataset(root, transforms)
    elif dset == 'droneface':
        dataset = DroneFaceDataset(root, ann_path=ann_path, transforms=transforms)
    else:
        dataset = WIDERFaceDataset(root, annotations=ann_path, transforms=transforms, split=split)
    return dataset


def get_loaders(args: Any) -> Tuple[DataLoader, DataLoader]:
    # By default, no augmentation
    # Currently, only flips are supported when applying augmentation
    augment = None
    if args.aug:
        augment = T_.Compose([
            T_.RandomHorizontalFlip(0.5),
            AddNoise(var=args.noise_var, mean=args.noise_mean)
        ])

    # Get the dataset
    if args.fudan:
        dataset = get_dataset(args.data_dir, transforms=augment)
    elif args.wider:
        dataset = get_dataset(args.data_dir, ann_path=args.ann_path, transforms=augment, dset='wider')
        val_dataset = get_dataset(args.data_dir, ann_path=args.ann_path, transforms=augment, dset='wider', split='val')
    else:
        dataset = get_dataset(args.data_dir, ann_path=args.ann_path, transforms=augment, dset='droneface')    
    
    
    if args.wider:
        train_data, val_data = dataset[:( args.num_batches*args.batch_size )], val_dataset
    
    else:
        # Create a 10:1 split on training/val data
        val_batches = (args.num_batches // 11) * args.batch_size
        train_batches = (args.num_batches - (args.num_batches // 11) ) * args.batch_size    
        train_data, val_data, _ = torch.utils.data.random_split(
            dataset, [train_batches, val_batches, len(dataset) - args.batch_size*args.num_batches]
        )

    # Create dataloaders, training data is shuffled
    # TODO: Add code to implement cross-validation
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn)

    # Return the training and validation loaders
    return train_loader, val_loader


def collate_fn(batch) -> Tuple:
    return tuple(zip(*batch))


def visualize_example(idx: int, weights: Optional[str] = None):
    # Load model and set to eval mode for inference
    model = SSDLite(2)
    model.eval()
    
    # Load weights if provided
    if weights is not None:
        model.load_state_dict( torch.load(weights, map_location='cpu') )
    
    # Compose transform for adding noise
    # augment = None
    augment = T_.Compose([
        AddNoise(0.2, 0.2)
    ])

    # Generate a noise-free sample for visualization and
    # the noisy sample for inference
    dataset = PennFudanDataset('data/PennFudanPed.zip', augment)
    d_ = PennFudanDataset('data/PennFudanPed.zip')

    # dataset = DroneFaceDataset('data/droneface.zip', 'data/droneface/annotations.csv', augment)
    # d_ = DroneFaceDataset('data/droneface.zip', 'data/droneface/annotations.csv')

    i_, _ = d_[idx]
    img, targets = dataset[idx]

    # Generate predictions
    with torch.no_grad():
        preds = model( img.unsqueeze(0) )

    img = img.permute(1,2,0)

    plt.figure()
    plt.imshow( i_.permute(1,2,0) )
    plt.axis('off')

    plt.figure()
    pil_img = Image.fromarray( ( img*255 ).numpy().astype(np.uint8) )

    # Get predicted boxes and initialize PIL ImageDraw object
    boxes = preds[0]['boxes']
    print(preds[0]['scores'])
    draw = ImageDraw.Draw(pil_img)

    # Count the number of valid boxes
    count = torch.sum( torch.where(preds[0]['scores'] > 0.4, 1, 0) )
    
    # Draw GT boxes 
    for box in targets['boxes']:
        draw.rectangle( tuple(box), width=3, outline='red' )

    # Draw predicted boxes
    for box in boxes[:count]:
        draw.rectangle( tuple(box), width=3, outline='green' )

    plt.imshow( pil_img )
    plt.axis('off')
    plt.show()



if __name__ == '__main__':
    torch.manual_seed(2023)
    parser = argparse.ArgumentParser()
    parser.add_argument('--idx', type=int, default=0)
    args = parser.parse_args()
    visualize_example(args.idx, 'data/vhigh.pth')
