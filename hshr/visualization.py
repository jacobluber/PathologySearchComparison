import argparse
import os

import openslide
import torch
import numpy as np

from preprocess import set_patch_size, backbone_model, extract_ft
from utils.data_utils import get_files_type
from utils.model.base_model import AttenHashEncoder
from utils.sampling import sample_patch_coors
from utils.sampling.sample_patches import get_just_gt_level
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def heatMap_color(score):
    assert 0 <= score <= 1
    color_list = [[0, 0, 255], [255, 255, 0], [225, 0, 0]]
    part_len = 1 / (len(color_list) - 1)
    part_idx = int(score / part_len)
    if part_idx == len(color_list) - 1:
        part_idx -= 1
    part_score = (score - part_idx * part_len) / part_len
    color = (1 - part_score) * np.array(color_list[part_idx]) + part_score * np.array(color_list[part_idx + 1])
    return color


def heatMap_img(slide, patch_coords, colors, alpha):
    mini_frac = 32
    mini_size = np.ceil(np.array(slide.level_dimensions[0]) / mini_frac).astype(int)
    mini_level = get_just_gt_level(slide, mini_size)
    img = slide.read_region((0, 0), mini_level, slide.level_dimensions[mini_level]).convert('RGB')
    img = img.resize(mini_size)
    ori = img
    img = np.asarray(img)
    for _coor, color in zip(patch_coords, colors):
        _mini_coor = (int(_coor[0] / mini_frac), int(_coor[1] / mini_frac))
        _mini_patch_size = (int(_coor[2] / mini_frac), int(_coor[3] / mini_frac))
        img[_mini_coor[1]:_mini_coor[1] + _mini_patch_size[1],
        _mini_coor[0]:_mini_coor[0] + _mini_patch_size[0]] = alpha * img[
                                                                     _mini_coor[1]:_mini_coor[1] + _mini_patch_size[1],
                                                                     _mini_coor[0]:_mini_coor[0] + _mini_patch_size[
                                                                         0]] + (
                                                                     1 - alpha) * color
    return Image.fromarray(img), ori


def adjust(att):
    att *= 1.5
    att[att > 1] = 1
    return att


def generate(svs_file, save_to, model_path):
    patch_size = 256
    color_min = 0.8
    alpha = 0.1
    img_save = os.path.join(save_to, svs_file.split('/')[-1][:-4])
    attention_model_path = model_path

    if not os.path.exists(img_save):
        os.makedirs(img_save)

    slide = openslide.open_slide(svs_file)
    patch_size = set_patch_size(slide, patch_size)
    coordinates, bg_mask = sample_patch_coors(slide, num_sample=-1, patch_size=patch_size, color_min=color_min,
                                              dense=True)
    print(len(coordinates))

    model, input_img_size, normalize = backbone_model(cnn_base='resnet')
    features = extract_ft(slide, model_ft=model, input_img_size=input_img_size, normalize=normalize,
                          patch_coors=coordinates, batch_size=256)
    model = AttenHashEncoder(512, 1024, 1)
    model.load_state_dict(torch.load(attention_model_path, map_location=device))
    model = model.attention_layer
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        att = model(features).cpu().detach().numpy()
    att = np.exp(att).reshape(-1)
    att = (att - att.min()) / (att.max() - att.min())
    att = adjust(att)
    colors = [heatMap_color(s) for s in att]

    img, ori = heatMap_img(slide, coordinates, colors, alpha)
    img.save(os.path.join(img_save, 'img.jpg'))
    ori.save(os.path.join(img_save, 'ori.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Preprocess raw WSIs")
    parser.add_argument("--SVS_DIR", type=str, required=True, help="The path of your WSI datasets.")
    parser.add_argument("--TMP", type=str, required=True, help="The path to save some necessary tmp files.")
    parser.add_argument("--SAVE_TO", type=str, required=True, help="The path to save visualization results")
    parser.add_argument("--MODEL_PATH", type=str, required=True, help="The path of trained CaEncoder")
    args = parser.parse_args()

    svss = get_files_type(args.SVS_DIR, 'svs', args.TMP)
    for svs in svss[20000:20020]:
        path = os.path.join(args.SVS_DIR, svs)
        generate(path, args.SAVE_TO, args.MODEL_PATH)
