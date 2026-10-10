# image_preprocessor.py
import numpy as np
import torch
from typing import Tuple

class ImagePreprocessor:
    def __init__(self, target_resolution: Tuple[int, int]):
        self.target_resolution = target_resolution

    def resize_and_pad(self, frame: np.ndarray) -> np.ndarray:
        """Resizes the frame while maintaining aspect ratio."""
        pass

    def to_tensor(self, processed_frame: np.ndarray) -> torch.Tensor:
        """Converts the processed numpy array to a normalized PyTorch tensor."""
        pass
