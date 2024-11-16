import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class EcommerceDataset(Dataset):
    def __init__(self, root_folder = None, image_folder = None, split='train',image_size=224):
        super(EcommerceDataset, self).__init__()
        self.root_folder = root_folder
        self.image_folder = image_folder
        self.split = split
        self.image_size = image_size
        self.info_file = os.path.join(root_folder, 'data.csv')
        self.df = pd.read_csv(self.info_file)
        self.df = self.df[self.df['split']==self.split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = {}
        image_path = row['path']
        item['image'] = Image.open(f"{self.image_folder}/{image_path}").convert('RGB').resize((self.image_size, self.image_size))
        item['text'] = row['product_name']
        item['labels'] = row['label']

        return item
