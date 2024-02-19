import os.path as osp
from itertools import product
from random import shuffle

import numpy as np
import openslide
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, star

BACKGROUND = 0
FOREGROUND = 1


def sample_patch_coors(slide, num_sample=2000, patch_size=256, color_min=0.8, dense=False):
    if dense:
        return dense_patch_coors(slide, patch_size, color_min)

    mini_frac = 32
    mini_size = np.ceil(np.array(slide.level_dimensions[0]) / mini_frac).astype(np.int64)
    mini_level = get_just_gt_level(slide, mini_size)
    mini_patch_size = patch_size // mini_frac

    if mini_level == 0:
        raise Exception('Image too large')
    try:
        bg_mask = generate_background_mask(slide, mini_level, mini_size)
    except MemoryError as e:
        slide.close()
        raise Exception('Handled Memory Error')

    assert bg_mask.shape == (mini_size[1], mini_size[0])

    # extract patches from available area
    patch_coors = []
    num_row, num_col = bg_mask.shape
    num_row = num_row - mini_patch_size
    num_col = num_col - mini_patch_size

    row_col = list(product(range(num_row), range(num_col)))
    shuffle(row_col)
    cnt = 0

    # attention center
    H_min = int(np.ceil(mini_patch_size / 8))
    H_max = int(np.ceil(mini_patch_size / 8 * 7))
    W_min = int(np.ceil(mini_patch_size / 8))
    W_max = int(np.ceil(mini_patch_size / 8 * 7))
    # half of the center
    th_num = int(np.ceil((mini_patch_size * 3 / 4 * mini_patch_size * 3 / 4)))

    for row, col in row_col:
        if cnt >= num_sample:
            break
        mini_patch = bg_mask[row:row + mini_patch_size, col: col + mini_patch_size]
        origin = (int(col * mini_frac), int(row * mini_frac), patch_size, patch_size)
        # print(np.count_nonzero(mini_patch[H_min:H_max, W_min:W_max]), th_num)
        # print(mini_patch)
        # print(H_min, H_max, W_min, W_max)
        if np.count_nonzero(mini_patch[H_min:H_max, W_min:W_max]) >= th_num * color_min:
            # filter those white background
            # if is_bg(slide, origin, patch_size):
            #     continue
            patch_coors.append(origin)
            cnt += 1

    return patch_coors, bg_mask


# get the just size that equal to mask_size
def get_just_gt_level(slide: openslide, size):
    level = slide.level_count - 1
    while level >= 0 and slide.level_dimensions[level][0] < size[0] and \
            slide.level_dimensions[level][1] < size[1]:
        level -= 1
    return level


def generate_background_mask(slide: openslide, mini_level, mini_size):
    img = slide.read_region((0, 0), mini_level, slide.level_dimensions[mini_level])
    img = img.resize(mini_size)
    bg_mask = threshold_segmentation(img)
    img.close()
    return bg_mask


# background segmentation algorithm
def threshold_segmentation(img):
    # calculate the overview level size and retrieve the image
    img_hsv = img.convert('HSV')
    img_hsv_np = np.array(img_hsv)

    # dilate image and then threshold the image
    schannel = img_hsv_np[:, :, 1]
    mask = np.zeros(schannel.shape)

    schannel = dilation(schannel, star(3))
    schannel = ndimage.gaussian_filter(schannel, sigma=(1, 1), order=0)
    threshold_global = threshold_otsu(schannel)

    mask[schannel > threshold_global] = FOREGROUND
    mask[schannel <= threshold_global] = BACKGROUND

    return mask


def is_bg(slide, origin, patch_size):
    img = slide.read_region(origin, 0, (patch_size, patch_size))
    # bad case is background
    if np.array(img)[:, :, 1].mean() > 200:  # is bg
        img.close()
        return True
    else:
        img.close()
        return False


def dense_patch_coors(slide, patch_size=256, color_min=0.8):
    mini_frac = 32
    mini_size = np.ceil(np.array(slide.level_dimensions[0]) / mini_frac).astype(np.int64)
    mini_level = get_just_gt_level(slide, mini_size)
    mini_patch_size = patch_size // mini_frac

    if mini_level == 0:
        raise Exception('Image too large')
    try:
        bg_mask = generate_background_mask(slide, mini_level, mini_size)
    except MemoryError as e:
        slide.close()
        raise Exception('Handled Memory Error')

    assert bg_mask.shape == (mini_size[1], mini_size[0])

    # extract patches from available area
    patch_coors = []
    num_row, num_col = bg_mask.shape
    num_row = num_row - mini_patch_size
    num_col = num_col - mini_patch_size

    row_col = list(product(range(0, num_row, mini_patch_size), range(0, num_col, mini_patch_size)))

    # attention center
    H_min = int(np.ceil(mini_patch_size / 8))
    H_max = int(np.ceil(mini_patch_size / 8 * 7))
    W_min = int(np.ceil(mini_patch_size / 8))
    W_max = int(np.ceil(mini_patch_size / 8 * 7))
    # half of the center
    th_num = int(np.ceil((mini_patch_size * 3 / 4 * mini_patch_size * 3 / 4)))

    for row, col in row_col:
        mini_patch = bg_mask[row:row + mini_patch_size, col: col + mini_patch_size]
        origin = (int(col * mini_frac), int(row * mini_frac), patch_size, patch_size)
        if np.count_nonzero(mini_patch[H_min:H_max, W_min:W_max]) >= th_num * color_min:
            patch_coors.append(origin)

    return patch_coors, bg_mask