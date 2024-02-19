import argparse
from os.path import basename, splitext, abspath, join

import h5py
from PIL import Image
from tqdm import tqdm
import numpy as np


Image.MAX_IMAGE_PIXELS = None


def combine_h5_files(input_paths, output_path):
    output_file = h5py.File(output_path, 'a')

    for input_path in tqdm(input_paths):
        i = int(splitext(basename(input_path))[0][-1])
        input_file = h5py.File(input_path, 'r')
        
        for key in input_file.keys():
            input_data = input_file[key][...]
            new_input_data = []
            for x, y in input_data:
                # since we have divided the image in 4 parts
                new_input_data.append(np.array([x + i * 21671, y]))
            new_input_data = np.array(new_input_data)
            data_shape = new_input_data.shape
            
            if key in output_file:
                dset = output_file[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                dset[-data_shape[0]:] = new_input_data
            else:
                data_type = new_input_data.dtype
                chunk_shape = (1, ) + data_shape[1:]
                maxshape = (None, ) + data_shape[1:]
                dset = output_file.create_dataset(key, shape=data_shape, maxshape=maxshape, chunks=chunk_shape, dtype=data_type)
                dset[:] = new_input_data
                
            for attr_key, attr_val in input_file[key].attrs.items():
                dset.attrs[attr_key] = attr_val
        
        input_file.close()

    output_file.close()
    return output_path

parser = argparse.ArgumentParser(description='For UCLA/slide2.svs, we have to split the slide and then merge it back together. This script merges the .h5 file back together.')
parser.add_argument('--splitted_slides_dir', type=str, default='./FEATURES/TEST_DATA/UCLA/patches',
                    help='path to folder containing splitted .h5 files.')


if __name__ == "__main__":
    args = parser.parse_args()
    splitted_slides_dir = abspath(args.splitted_slides_dir)

    file_list = [join(splitted_slides_dir, f"slide2_{i}.h5") for i in range(4)]
    output = join(splitted_slides_dir, "slide2.h5")

    combine_h5_files(file_list, output)

