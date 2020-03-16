from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import warnings
warnings.filterwarnings("ignore")

plt.ion()
csv_file = './frames/frames.csv'

class UCF11Dataset(Dataset):
  def __init__(self, csv_file, root_dir, transform=None):
    self.frames = pd.read_csv(csv_file)
    self.root_dir = root_dir

  def __len__(self):
    return len(self.frames)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    frames = self.frames.iloc[idx, 0].split('|')
    frames = frames[:-1]
    cat = self.frames.iloc[idx, 1]
    frame_list = []
    for f in frames:
      frame_name = os.path.join(self.root_dir, f)
      frame = io.imread(frame_name)
      frame = transform.resize(frame, (128, 128))
      frame_list.append(frame)
    sample = {'frame' : frame_list, 'category' : cat}
    return sample


class ToTensor(object):
  def __call__(self, sample):
    frame, label = sample['frame'], sample['category']
    frame = frame.permute((2, 0, 1))
    return {'frame' : torch.from_numpy(frame), 'category': label}