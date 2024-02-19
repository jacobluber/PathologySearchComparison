# -*- coding: utf-8 -*-
"""
@Time    : 2021/9/17 16:18
@Author  : Lucius
@FileName: call.py
@Software: PyCharm
"""
import self_supervision.moco.builder


def get_moco(encoder_q, encoder_k, device, n_target):
    K = 65536
    m = 0.999
    T = 0.07
    return self_supervision.moco.builder.MoCo(
        encoder_q.to(device),
        encoder_k.to(device),
        device,
        n_target, K, m, T, False)

