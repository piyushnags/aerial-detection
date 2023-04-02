# Built-in Imports
import os
from typing import Callable, Optional, Tuple, List, Dict
from io import BytesIO
from zipfile import ZipFile

# Image Processing and CV Imports
import cv2
from PIL import Image
import numpy as np

# PyTorch Imports
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T


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
        
        targets['boxes'] = boxes
        targets['labels'] = labels
        targets['scores'] = scores

        if self.transforms is not None:
            img, targets = self.transforms(img, targets)
        
        return img, targets
    

    def __len__(self) -> int:
        return len(self.img_list)

   