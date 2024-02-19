# -*- coding: utf-8 -*-
"""
@Time    : 2022/1/11 13:29
@Author  : Lucius
@FileName: retrieval_utils.py
@Software: PyCharm
"""

import torch
import numpy as np


def generate_incidence(key_list, top_idx, patch_num, k, patch_weights=None):
    if patch_weights is not None:
        assert len(key_list) == patch_weights.shape[0]
    list_slide_id = []
    for idx, key in enumerate(key_list):
        if idx % patch_num == 0:
            slide_id = key.split('@')[-2]
            list_slide_id.append(slide_id)
        else:
            assert key_list[idx - 1].split('@')[-2] == key.split('@')[-2]

    slide_size = len(key_list) // patch_num
    incidence = np.zeros([slide_size, slide_size])
    logk = np.log(k + 1)
    for q_id, top in enumerate(top_idx):
        for i in range(k):
            patch_id = top[i]
            slide_id = patch_id // patch_num
            q_slide_id = q_id // patch_num
            w = 1 if patch_weights is None else patch_weights[patch_id]
            incidence[q_slide_id][slide_id] += (1 - (np.log(i + 1) / logk)) * w
    # normalize
    degree = incidence[0].sum()
    incidence = incidence / degree
    return incidence, list_slide_id


def generate_sh(inc):
    return inc.dot(np.transpose(inc))


def generate_sv(inc):
    return np.transpose(inc).dot(inc)


def normalize(mat):
    row_sums = mat.sum(axis=1)
    new_matrix = mat / row_sums[:, np.newaxis]
    return new_matrix


def hyedge_similarity(inc, alpha, beta):
    sh = generate_sh(inc)
    sv = generate_sv(inc)
    ss = inc + inc.T
    sh, sv, ss = normalize(sh), normalize(sv), normalize(ss)

    simi = sh + alpha * sv + beta * ss
    _, slide_top_idx = torch.topk(torch.from_numpy(simi), simi.shape[0], dim=1, largest=True)

    return slide_top_idx
