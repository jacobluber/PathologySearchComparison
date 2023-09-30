# By Abtin Riasatian, email: abtin.riasatian@uwaterloo.ca

# The extract_features function gets a patch directory and a feature directory.
# the function will extract the features of the patches inside the folder
# and saves them in a pickle file of dictionary mapping patch names to features.
# =============================================================

import os
from os.path import join, abspath
import glob
import pickle
import pathlib
import argparse

import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Lambda
# from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.backend import bias_add, constant    
from PIL import Image
import numpy as np
from tqdm import tqdm


# feature extractor preprocessing function
def preprocessing_fn(inp, sz=(1000, 1000)):
    out = tf.cast(inp, 'float') / 255.    
    out = tf.cond(tf.equal(tf.shape(inp)[1], sz[0]), lambda: out, lambda: tf.image.resize(out, sz))
    mean = tf.reshape((0.485, 0.456, 0.406), [1, 1, 1, 3])
    std = tf.reshape((0.229, 0.224, 0.225), [1, 1, 1, 3])
    out = out - mean
    out = out / std
    return out

def get_dn121_model():
    model = tf.keras.applications.densenet.DenseNet121(input_shape=(1000, 1000, 3), include_top=False,pooling='avg')
    seq_model = tf.keras.models.Sequential([tf.keras.layers.Lambda(preprocessing_fn, input_shape=(None, None, 3), dtype=tf.uint8)])
    seq_model.add(model)
    return seq_model

# feature extraction function
def extract_features(patch_dir, extracted_features_save_adr, experiment, network_weights_address, network_input_patch_width, batch_size, img_format):
    feature_extractor = get_dn121_model()
    if experiment == "ABLATION_DATABASE":
        all_patches = join(patch_dir, '*', '*', '*', '*.' + img_format)
    elif experiment in ["UCLA", "READER_STUDY", "GBM_MICROSCOPE_CPTAC", "GBM_MICROSCOPE_UPENN", "BRCA_HER2", "BRCA_TRASTUZUMAB", "ABLATION_BRCA_TRASTUZUMAB"]:
        all_patches = join(patch_dir, '*', '*.' + img_format)
    patch_adr_list = [pathlib.Path(x) for x in glob.glob(all_patches)]
    feature_dict = {}
    for batch_st_ind in tqdm(range(0, len(patch_adr_list), batch_size)):
        batch_end_ind = min(batch_st_ind + batch_size, len(patch_adr_list))
        batch_patch_adr_list = patch_adr_list[batch_st_ind:batch_end_ind]
        target_shape = (network_input_patch_width, network_input_patch_width)
        patch_batch = np.array([np.array(Image.open(x).resize(target_shape)) for x in batch_patch_adr_list])
        batch_features = feature_extractor.predict(patch_batch)
        feature_dict.update(dict(zip([x.parents[0].name + "_" + x.name for x in batch_patch_adr_list], list(batch_features))))
        with open(extracted_features_save_adr, 'wb') as output_file:
            pickle.dump(feature_dict, output_file, pickle.HIGHEST_PROTOCOL)


parser = argparse.ArgumentParser(description="KimiaNet feature extraction")
parser.add_argument('--patch_dir', default='./PATCHES/', help='Directory of patches')
parser.add_argument('--extracted_features_save_adr', default='./extracted_features.pkl', help='Address to save extracted features')
parser.add_argument('--network_weights_address', default='./checkpoints/KimiaNetKerasWeights.h5', help='Address of network weights')
parser.add_argument("--experiment", type=str, default="DATABASE", help="Can be in ['DATABASE', 'UCLA', 'READER_STudy'].")
parser.add_argument('--network_input_patch_width', type=int, default=1000, help='Width of network input patch')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
parser.add_argument('--img_format', default='png', help='Patch image format')
parser.add_argument('--use_gpu', type=bool, default=True, help='Whether to use GPU')


if __name__ == "__main__":
    args = parser.parse_args()

    if args.use_gpu:
        os.environ['NVIDIA_VISIBLE_DEVICES'] = '0,1'
        os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    extract_features(abspath(args.patch_dir), abspath(args.extracted_features_save_adr), args.experiment, args.network_weights_address, args.network_input_patch_width, args.batch_size, args.img_format)
    