# -*- coding: utf-8 -*-
"""
@Time    : 2021/6/11 11:27
@Author  : Lucius
@FileName: preprocess.py
@Software: PyCharm
"""
import argparse
import random

import openslide
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import os
import numpy as np

import sys
sys.path.append("../")

from utils.sampling import sample_patch_coors
from utils.data_utils import *


def generate_patch(slide, patch_coors):
    patches = []
    for i, coor in enumerate(patch_coors):
        img = slide.read_region((coor[0], coor[1]), 0, (coor[2], coor[3])).convert('RGB')
        patches.append(img)
    return patches


def set_patch_size(slide, patch_size):
    p = slide.properties
    mag = p['aperio.AppMag']
    if mag != '20':
        patch_size *= 2
    return patch_size


def generate_slide(svs_dir, result_dir, tmp_path, size):
    svs_relative_path_list = get_files_type(svs_dir, 'svs', tmp_path)
    random.seed(0)
    random.shuffle(svs_relative_path_list)
    svs_relative_path_list = svs_relative_path_list[:size]

    for idx, svs_relative_path in enumerate(tqdm(svs_relative_path_list)):
        file_dir = os.path.join(result_dir, 'slide_{}'.format(idx))
        if os.path.exists(file_dir):
            continue
        svs_file = os.path.join(svs_dir, svs_relative_path)
        try:
            slide = openslide.open_slide(svs_file)
            patch_size = set_patch_size(slide, 256)
            coordinates, bg_mask = sample_patch_coors(slide, num_sample=100, patch_size=patch_size, color_min=0.8)
            patches = generate_patch(slide, coordinates)
            file_dir = os.path.join(result_dir, 'slide_{}'.format(idx))
            os.makedirs(file_dir)
            assert len(patches) == 100
            for p_idx, p in enumerate(patches):
                p.save(os.path.join(file_dir, '{}.jpg'.format(p_idx)))

        except MemoryError as e:
            print('While handling ', svs_relative_path)
            print("find Memory Error, exit")
            exit()
        except Exception as e:
            print(e)
            print("failing in one image, continue")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocess raw WSIs")
    parser.add_argument("--SVS_DIR", type=str, required=True, help="The path of your WSI datasets.")
    parser.add_argument("--RESULT_DIR", type=str, required=True, help="A path to save your results.")
    parser.add_argument("--TMP", type=str, required=True, help="The path to save some necessary tmp files.")
    parser.add_argument("--SIZE", type=int, default=1000, help="size of slides")
    args = parser.parse_args()

    # slide = openslide.open_slide('/home2/lishengrui/new_exp/HSHR/WSI/paad/TCGA-2J-AAB8-01A-01-TSA.svs')
    generate_slide(args.SVS_DIR, args.RESULT_DIR, args.TMP, args.SIZE)
    # python generate.py --SVS_DIR /home2/lishengrui/new_exp/HSHR/WSI --RESULT_DIR /home2/lishengrui/major1026/backbone/dataset --TMP /home2/lishengrui/new_exp/HSHR/TMP
