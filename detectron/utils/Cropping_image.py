import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import os
import glob


class MyCrop:
    """Crop"""

    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, x):
        return TF.crop(x, self.top, self.left, self.height, self.width)
