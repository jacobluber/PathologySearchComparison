# -*- coding: utf-8 -*-
"""
@Time    : 2022/4/11 16:49
@Author  : Lucius
@FileName: feature.py
@Software: PyCharm
"""

import os
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm
import pickle
from utils.data_utils import get_files_type


def mean_feature(feature_and_coordinate_dir, tmp, data_from=0, data_to=1):
    p = os.path.join(tmp, 'ssl_mean_feature')
    p0 = p + '0'
    p1 = p + '1'
    if os.path.exists(p0 + '.npy'):
        means_0 = np.load(p0 + '.npy')
        means_1 = np.load(p1 + '.npy')
        with open(p + '.pkl', 'rb') as f:
            paths = pickle.load(f)
        print('load cache')
        return means_0, means_1, paths
    else:
        means_0 = list()
        means_1 = list()
        paths = list()
        feature_list = get_files_type(feature_and_coordinate_dir, '0.npy', tmp)
        feature_list.sort()
        size = len(feature_list)
        for feature_path in tqdm(feature_list[int(data_from * size):int(data_to * size)]):
            base_name = os.path.basename(feature_path)
            dir_name = os.path.join(feature_and_coordinate_dir, os.path.dirname(feature_path))
            if base_name == '0.npy':
                files = os.listdir(dir_name)
                if '1.npy' in files and '0.pkl' in files and '1.pkl' in files:
                    paths.append(os.path.dirname(feature_path))

                    npy_file_0 = os.path.join(dir_name, '0.npy')
                    x0 = np.load(npy_file_0)
                    mean_0 = np.mean(x0, axis=0)
                    means_0.append(mean_0)

                    npy_file_1 = os.path.join(dir_name, '1.npy')
                    x1 = np.load(npy_file_1)
                    mean_1 = np.mean(x1, axis=0)
                    means_1.append(mean_1)

        means_0 = np.array(means_0)
        means_1 = np.array(means_1)
        np.save(p0, means_0)
        np.save(p1, means_1)
        with open(p + '.pkl', 'wb') as fp:
            pickle.dump(paths, fp)
        return means_0, means_1, paths


def cluster_feature(feature_and_coordinate_dir, tmp, types, num_cluster=20, data_from=0, data_to=1):
    p = os.path.join(tmp, "&".join(types)+'_'+str(num_cluster))
    if os.path.exists(p + '.npy'):
        clusters = np.load(p + '.npy')
        with open(p + '.pkl', 'rb') as f:
            paths = pickle.load(f)
        return clusters, paths
    else:
        clusters = list()
        paths = list()
        feature_list = get_files_type(feature_and_coordinate_dir, '0.npy', tmp)
        feature_list.sort()
        size = len(feature_list)
        for feature_path in tqdm(feature_list[int(data_from * size):int(data_to * size)]):
            base_name = os.path.basename(feature_path)
            dir_name = os.path.join(feature_and_coordinate_dir, os.path.dirname(feature_path))
            if base_name == '0.npy':
                files = os.listdir(dir_name)
                if '0.pkl' in files:
                    if dir_name.split('/')[-2] in types:
                        paths.append(os.path.dirname(feature_path))
                        npy_file_0 = os.path.join(dir_name, '0.npy')
                        x0 = np.load(npy_file_0)
                        while len(x0.shape) > 2:
                            x0 = x0.squeeze()
                        km = KMeans(n_clusters=num_cluster)
                        km.fit(x0)
                        clusters.append(km.cluster_centers_)
        clusters = np.array(clusters)

        np.save(p, clusters)
        with open(p + '.pkl', 'wb') as fp:
            pickle.dump(paths, fp)
        return clusters, paths


def cluster_reduce(features, num_cluster):
    """

    Args:
        features: size x dim
        num_cluster: number of clusters

    Returns:
        cluster_center: num_cluster x dim
    """
    while len(features.shape) > 2:
        features = features.squeeze()
    km = KMeans(n_clusters=num_cluster, n_init=10)
    km.fit(features)
    return km.cluster_centers_


def min_max_binarized(feats):
    input_shape = feats.shape
    dim = feats.shape[-1]
    feats = feats.reshape(-1, dim)
    prev = np.concatenate([np.zeros([feats.shape[0], 1]), feats[:, :-1]], axis=1)
    h = np.sign(feats-prev)
    return h.reshape(input_shape)

    # input_shape = feats.shape
    # all_output = []
    # for feat in tqdm(feats):
    #     prev = float('inf')
    #     output_binarized = []
    #     for ele in feat:
    #         if ele < prev:
    #             code = -1
    #             output_binarized.append(code)
    #         elif ele >= prev:
    #             code = 1
    #             output_binarized.append(code)
    #         prev = ele
    #     all_output.append(output_binarized)
    # all_output = np.array(all_output)
    # all_output = all_output.reshape(input_shape)
    # return all_output