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
def preprocessing_fn(input_batch, network_input_patch_width):
    org_input_size = tf.shape(input_batch)[1]
    # standardization
    scaled_input_batch = tf.cast(input_batch, 'float') / 255.
    # resizing the patches if necessary
    resized_input_batch = tf.cond(tf.equal(org_input_size, network_input_patch_width), 
                                  lambda: scaled_input_batch, 
                                  lambda: tf.image.resize(scaled_input_batch, (network_input_patch_width, network_input_patch_width)))
    # normalization, this is equal to tf.keras.applications.densenet.preprocess_input()---------------
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    data_format = "channels_last"
    mean_tensor = constant(-np.array(mean))
    standardized_input_batch = bias_add(resized_input_batch, mean_tensor, data_format)
    standardized_input_batch /= std
    return standardized_input_batch

# feature extractor initialization function
def kimianet_feature_extractor(network_input_patch_width, weights_address):
    dnx = DenseNet121(include_top=False, weights=weights_address, input_shape=(network_input_patch_width, network_input_patch_width, 3), pooling='avg')
    kn_feature_extractor = Model(inputs=dnx.input, outputs=GlobalAveragePooling2D()(dnx.layers[-3].output))
    kn_feature_extractor_seq = Sequential([Lambda(preprocessing_fn, arguments={'network_input_patch_width': network_input_patch_width}, input_shape=(None, None, 3), dtype=tf.uint8)])
    kn_feature_extractor_seq.add(kn_feature_extractor)
    return kn_feature_extractor_seq

# feature extraction function
def extract_features(patch_dir, extracted_features_save_adr, experiment, network_weights_address, network_input_patch_width, batch_size, img_format):
    feature_extractor = kimianet_feature_extractor(network_input_patch_width, network_weights_address)
    if experiment == "DATABASE":
        all_patches = join(patch_dir, '*', '*', '*', '*.' + img_format)
    elif experiment in ["UCLA", "READER_STUDY", "GBM_MICROSCOPE_CPTAC", "GBM_MICROSCOPE_UPENN", "BRCA_HER2", "BRCA_TRASTUZUMAB"]:
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
    