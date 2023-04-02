# Built-in Imports
from typing import List, Any, Optional, Dict, Tuple

# PyTorch Imports
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.detection import(
    ssdlite320_mobilenet_v3_large, 
    SSDLite320_MobileNet_V3_Large_Weights
)



class SSDLite(nn.Module):
    '''
    Description:
        Wrapper for SSDLite MobileNetv3. Modified the model
        to support VisDrone dataset as described in
        https://github.com/VisDrone/VisDrone2018-DET-toolkit

        PyTorch implementation of SSDLite MobileNetV3 is based 
        on the implementation described in the "Searching for MobileNetV3"
        paper: https://arxiv.org/abs/1905.02244
    '''
    def __init__(self, num_classes: int, pretrained: bool = False):
        super(SSDLite, self).__init__()
        
        if pretrained:
            model = ssdlite320_mobilenet_v3_large(
                num_classes=num_classes, 
                weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
            )
        else:
            model = ssdlite320_mobilenet_v3_large(
                num_classes=num_classes
            )
        
        self.model = model
    

    def forward(self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
                ) -> Tuple[ Dict[str, Tensor], List[Dict[str, Tensor]] ]:
        '''
        Description: Forward implementation for the SSDLite wrapper
        Args: 
            images: batch of images in the form of torch tensors
                    (batch needs to be normalized before calling forward)
            targets: List of dictionaries containing confidence scores,
                     box coordinates, and labels during training, else
                     None.
        Returns:
            Dict containing losses during training
            Batch of predictions containing scores, labels, and coordinates
            See Also: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/ssd.py#L126
        '''
        return self.model(images, targets)
