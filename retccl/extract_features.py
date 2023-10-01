import argparse
from os.path import join, abspath
from os import makedirs
import glob
import json
import pickle

import numpy as np
import pandas as pd
import h5py
import openslide
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from model import ccl_model


class roi_dataset(Dataset):
    def __init__(self, patch_dataframe, transforms):
        super().__init__()
        self.patch_dataframe = patch_dataframe
        self.transforms = transforms

    def __len__(self):
        return len(self.patch_dataframe)

    def __getitem__(self, idx):
        path = self.patch_dataframe.loc[idx, "slide_path"]
        coord = (self.patch_dataframe.loc[idx, "coord1"], self.patch_dataframe.loc[idx, "coord2"])
        patch_level = self.patch_dataframe.loc[idx, "patch_level"]
        patch_size = self.patch_dataframe.loc[idx, "patch_size"]
        slide = openslide.open_slide(path)
        patch_region = slide.read_region(coord, patch_level, (patch_size, patch_size)).convert('RGB')
        patch_region = self.transforms(patch_region)
        return patch_region


parser = argparse.ArgumentParser(description="RetCCL feature extraction.")
parser.add_argument('--patch_dataframe_path', default='./FEATURES/DATABASE/patch_dataframe.csv', help='path to patch_dataframe.csv')
parser.add_argument('--save_dir', default='./FEATURES/DATABASE', help='directory to save temporary logs and final results.')


if __name__ == "__main__":
    args = parser.parse_args()
    patch_dataframe_path = abspath(args.patch_dataframe_path)
    save_dir = abspath(args.save_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    trnsforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    patch_dataframe = pd.read_csv(patch_dataframe_path)
    dataset = roi_dataset(patch_dataframe, trnsforms)
    database_loader = DataLoader(dataset, batch_size=1024, shuffle=False)
    
    ccl = ccl_model().to(device)
    ccl.eval()
    features_list = []
    i = 0
    with torch.no_grad():
        for batch in tqdm(database_loader):
            batch = batch.to(device)
            features = ccl(batch)
            features = features.cpu()
            path = join(save_dir, "_features_logs")
            makedirs(path, exist_ok=True)
            with open(join(path, f"features_{i}.pkl"), "wb") as f:
                pickle.dump(features, f)
            features_list.extend(features.numpy())
            i += 1
    
    patch_dataframe.loc[:, "features"] = features_list
    patch_dataframe.to_hdf(join(save_dir, "features.h5"), key="df", mode="w")
    
    print("feature extraction completed successfully.")