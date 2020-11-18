from dataset import TorchBIDS
import torch as T
from torch.utils.data import DataLoader
import numpy as np

class BIDSLoader(DataLoader):
    def __init__(self):