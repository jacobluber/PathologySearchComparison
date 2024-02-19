# -*- coding: utf-8 -*-
"""
@Time    : 2022/1/4 20:14
@Author  : Lucius
@FileName: patch_acc.py
@Software: PyCharm
"""
import numpy as np
import torch

from utils.emd_distance import getEMD


def patch_accuracy(key_list, top_idx, r):
    assert len(key_list) == top_idx.shape[0]
    slide_counter = {}
    c = 0
    for key in key_list:
        # class_truth = key.split('/')[-2]
        slide_id = key.split('@')[-2]
        # patch_id = key.split('/')[-1].split('_')[-1]
        if slide_id not in slide_counter.keys():
            slide_counter[slide_id] = c
            c += 1
    list_slide_id = list(slide_counter.keys())
    slide_num = len(list_slide_id)
    dis_matrix = np.zeros([slide_num, slide_num])
    for top in top_idx:
        cur_key = key_list[top[0]]
        cur_slide_id = cur_key.split('@')[-2]
        cur_id = slide_counter[cur_slide_id]
        for i in range(1, r):
            key = key_list[top[i]]
            slide_id = key.split('@')[-2]
            id = slide_counter[slide_id]
            dis_matrix[id][cur_id] += r - i
            dis_matrix[cur_id][id] += r - i
    _, slide_top_idx = torch.topk(torch.from_numpy(dis_matrix), dis_matrix.shape[0], dim=1, largest=True)

    return list_slide_id, slide_top_idx


def patch_accuracy_v2(key_list, top_idx, dis_mat):
    PATCH = 20
    dim = dis_mat.shape[0]
    mean_dis_mat = dis_mat.reshape(int(dim / PATCH), PATCH, int(dim / PATCH), PATCH).mean(3).mean(1)
    _, slide_top_idx = torch.topk(torch.from_numpy(mean_dis_mat), mean_dis_mat.shape[0], dim=1, largest=False)
    list_slide_id = []
    for idx, key in enumerate(key_list):
        if idx % PATCH == 0:
            slide_id = key.split('@')[-2]
            list_slide_id.append(slide_id)
        else:
            assert key_list[idx-1].split('@')[-2] == key.split('@')[-2]
    return list_slide_id, slide_top_idx


def patch_accuracy_v3(key_list, top_idx, dis_mat):
    PATCH = 20
    dim = dis_mat.shape[0]
    reshaped = dis_mat.reshape(int(dim / PATCH), PATCH, int(dim / PATCH), PATCH)
    score_mat = np.min(reshaped, axis=3)
    score_mat = np.median(score_mat, axis=1)
    _, slide_top_idx = torch.topk(torch.from_numpy(score_mat), score_mat.shape[0], dim=1, largest=False)
    list_slide_id = []
    for idx, key in enumerate(key_list):
        if idx % PATCH == 0:
            slide_id = key.split('@')[-2]
            list_slide_id.append(slide_id)
        else:
            assert key_list[idx-1].split('@')[-2] == key.split('@')[-2]
    return list_slide_id, slide_top_idx


def patch_accuracy_emd(key_list, top_idx, dis_mat, r=40):
    dim = dis_mat.shape[0]
    num_wsi = int(dim / 20)

    w = np.ones(20)
    w /= 20
    wsi_dis_mat = np.zeros([num_wsi, num_wsi])
    for i in range(num_wsi):
        for j in range(num_wsi):
            d = getEMD(dis_mat[i:i+20, j:j+20], 20, 20, w, w)
            wsi_dis_mat[i][j] = d

    _, slide_top_idx = torch.topk(torch.from_numpy(wsi_dis_mat), wsi_dis_mat.shape[0], dim=1, largest=False)
    list_slide_id = []
    for idx, key in enumerate(key_list):
        if idx % 20 == 0:
            slide_id = key.split('@')[-2]
            list_slide_id.append(slide_id)
        else:
            assert key_list[idx-1].split('@')[-2] == key.split('@')[-2]
    return list_slide_id, slide_top_idx