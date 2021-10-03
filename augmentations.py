from typing import Tuple

import torch
import torch.nn as nn
from kornia.augmentation import (
    ColorJitter,
    RandomGaussianBlur,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    VideoSequential,
)
from kornia.augmentation.container.video import VideoSequential
from torch.functional import Tensor

# agumentatin class for spatial and temporal augmentations as mentioned in paper


class DataAugmentation(nn.Module):
    """applies for spatial and temporal augmentations"""

    def __init__(
        self,
        crop_ratio: Tuple = (0.3, 1),
        aspect_ratio: Tuple = (0.5, 2),
        resize_size: int = 224,
        p_flip: float = 0.5,
        p_jitter: float = 0.8,
        p_grey: float = 0.2,
    ) -> None:
        super().__init__()
        self.crop_ratio = crop_ratio
        self.aspect_ratio = aspect_ratio
        self.resize = resize_size
        self.flip = p_flip
        self.jitter = p_jitter
        self.grey = p_grey

    def _apply_spatial_augmentation(self):
        spatial_aug = VideoSequential(
            RandomResizedCrop(
                size=(self.resize, self.resize),
                scale=self.crop_ratio,
                ratio=self.aspect_ratio,
            ),
            RandomHorizontalFlip(p=self.flip),
            ColorJitter(p=self.jitter),
            RandomGrayscale(p=self.grey),
            RandomGaussianBlur(p=1),
            data_format="BTCHW",
            same_on_frame=True,
        )

    def _apply_temporal_augmentation(self):
        pass

    def forward(self, input: Tensor) -> Tensor:
        pass
