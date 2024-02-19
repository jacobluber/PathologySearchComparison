# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/11 11:27
@Author  : Lucius
@FileName: preprocess.py
@Software: PyCharm
"""
import argparse

import openslide
import torch
from torchvision.models import densenet121
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from backbone.SimCLR.models.resnet_simclr import ResNetSimCLR
from utils.data_utils import *
from utils.feature import cluster_reduce
from utils.model.base_cnns import ResNetFeature, VGGFeature
import os
import pickle

from utils.model.base_model import SqueezeOp
from utils.sampling import sample_patch_coors
import numpy as np


SSL_RESNET_MODEL_DICT = "FEATURES/DATABASE/BACKBONE_RESULTS/checkpoint_199.pt"

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def backbone_model(depth=34, cnn_base='resnet'):
    if cnn_base == 'resnet':
        model_ft = ResNetFeature(depth=depth, pooling=True, pretrained=True)
        input_img_size = 224
    elif cnn_base == 'densenet':
        densenet = densenet121(pretrained=True)
        densenet = torch.nn.Sequential(*list(densenet.children())[:-1], torch.nn.AvgPool2d(kernel_size=(32, 32)))
        model_ft = densenet
        input_img_size = 1024
    elif cnn_base == 'ssl_resnet':
        ssl_resnet = ResNetSimCLR(base_model='resnet18', out_dim=128)
        ssl_resnet.load_state_dict(torch.load(SSL_RESNET_MODEL_DICT, map_location=device))
        model_ft = torch.nn.Sequential(*list(ssl_resnet.backbone.children())[:-1], SqueezeOp())
        input_img_size = 224
    else:
        model_ft = VGGFeature(depth=depth, pooling=True, pretrained=True)
        input_img_size = 224
    model_ft.eval()
    model_ft = model_ft.to(device)
    normalize = True
    if cnn_base == 'ssl_resnet':
        normalize = False
    return model_ft, input_img_size, normalize


def extract_ft(slide, model_ft, input_img_size, normalize, patch_coors, batch_size=128):
    dataset = Patches(slide, patch_coors, input_img_size, normalize)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    fts = []
    with torch.no_grad():
        for _patches in dataloader:
            _patches = torch.squeeze(_patches, 1)
            _patches = _patches.to(device, non_blocking=True)
            _fts = model_ft(_patches)
            fts.append(_fts)
    fts = torch.cat(fts, dim=0)
    assert fts.size(0) == len(patch_coors)
    return fts


class Patches(Dataset):
    def __init__(self, slide: openslide, patch_coors, input_img_size=224, normalize=True) -> None:
        super().__init__()
        self.slide = slide
        self.patch_coors = patch_coors
        if normalize:
            self.transform = transforms.Compose([
                transforms.Resize(input_img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(input_img_size),
                transforms.ToTensor(),
            ])

    def __getitem__(self, idx: int):
        coor = self.patch_coors[idx]
        img = self.slide.read_region((coor[0], coor[1]), 0, (coor[2], coor[3])).convert('RGB')
        return self.transform(img)

    def __len__(self) -> int:
        return len(self.patch_coors)


def set_patch_size(slide, patch_size):
    p = slide.properties
    mag = p['openslide.objective-power']
    if mag != '20':
        patch_size *= 2
    return patch_size


def handle_slide(slide, model_ft, input_img_size, normalize, num_sample=2000, patch_size=256, batch_size=256,
                 color_min=0.8, dense=False):
    patch_size = set_patch_size(slide, patch_size)
    coordinates, bg_mask = sample_patch_coors(slide, num_sample=num_sample, patch_size=patch_size, color_min=color_min,
                                              dense=dense)
    features = extract_ft(slide, model_ft=model_ft, input_img_size=input_img_size, normalize=normalize,
                          patch_coors=coordinates, batch_size=batch_size)
    return coordinates, features


def preprocess(wsi_dir, slide_ext, result_dir, tmp_path, dense, save_all):
    relative_path_list = get_files_type(wsi_dir, slide_ext, tmp_path)
    # relative_path_list = [p[1:] for p in relative_path_list]
    outputs = ['clu_0.npy', 'clu_1.npy']
    if save_all:
        outputs = ['0.pkl', '0.npy', '1.pkl', '1.npy']
    todo_list = check_todo(result_dir, relative_path_list, outputs)

    if SSL_RESNET_MODEL_DICT:
        model, input_img_size, normalize = backbone_model(cnn_base='ssl_resnet')  # ssl_resnet
    else:
        model, input_img_size, normalize = backbone_model(cnn_base='resnet')  # ssl_resnet

    for relative_path in tqdm(todo_list):
        wsi_file = os.path.join(wsi_dir, relative_path) + f".{slide_ext}"
        
        try:
            slide = openslide.open_slide(wsi_file)

            if not dense:
                for idx in ['0', '1']:
                    coordinates, features = handle_slide(slide, model, input_img_size, normalize)
                    clu_centers = cluster_reduce(features.cpu().numpy(), 20)
                    np.save(get_save_path(result_dir, relative_path, 'clu_{}.npy'.format(idx)), clu_centers)

                    if save_all:
                        coordinates_file = get_save_path(result_dir, relative_path, idx + '.pkl')
                        with open(coordinates_file, 'wb') as fp:
                            pickle.dump(coordinates, fp)
                        np.save(get_save_path(result_dir, relative_path, idx + '.npy'), features.cpu().numpy())
            else:
                coordinates, features = handle_slide(slide, model, input_img_size, normalize, dense=True)
                features = features.cpu().numpy()
                np.random.shuffle(features)
                fs = int(features.shape[0]/2)
                f0, f1 = features[:fs], features[fs:]
                c0, c1 = cluster_reduce(f0, 20), cluster_reduce(f1, 20)
                np.save(get_save_path(result_dir, relative_path, 'clu_0.npy'), c0)
                np.save(get_save_path(result_dir, relative_path, 'clu_1.npy'), c1)

        except MemoryError as e:
            print('While handling ', relative_path)
            print("find Memory Error, exit")
            exit()
        except Exception as e:
            print(e)
            print(f"failing in {relative_path} image, continue")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocess raw WSIs")
    parser.add_argument("--WSI_DIR", type=str, required=True, help="The path of your WSI datasets.")
    parser.add_argument("--SLIDE_EXT", default='svs', help="Slide file format")
    parser.add_argument("--RESULT_DIR", type=str, required=True, help="A path to save your preprocessed results.")
    parser.add_argument("--TMP", type=str, required=True, help="The path to save some necessary tmp files.")
    parser.add_argument("--DENSE", type=bool, default=True, help="densely extract patches")
    parser.add_argument("--SAVE_ALL", type=bool, default=True, help="also save features and coordinates")
    args = parser.parse_args()
    print(args)

    preprocess(args.WSI_DIR, args.SLIDE_EXT, args.RESULT_DIR, args.TMP, args.DENSE, args.SAVE_ALL)
