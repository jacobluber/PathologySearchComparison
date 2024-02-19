import argparse
from os import makedirs, listdir
from os.path import join, splitext, abspath

import h5py
from openslide import open_slide
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Visualizing every patches resulted from create_patches_fp and saving to file.')
parser.add_argument('--slides_dir', type=str, default='/raid/nejm_ai/TEST_DATA/READER_STUDY',
                    help='path to folder containing raw wsi image files.')
parser.add_argument('--patches_dir', type=str, default='./FEATURES/TEST_DATA/READER_STUDY/patches',
                    help='directory containing the created .h5 patches.')
parser.add_argument('--save_dir', type=str, default='./FEATURES/TEST_DATA/READER_STUDY/visualized_patches',
                    help='directory to save the visualized patches.')
parser.add_argument('--slide_extension', type=str, default='.svs',
                    help='slides extension. Example: svs or ndpi')

if __name__ == "__main__":
    args = parser.parse_args()

    slides_dir = abspath(args.slides_dir)
    patches_dir = abspath(args.patches_dir)
    save_dir = abspath(args.save_dir)
    slide_extension = args.slide_extension

    makedirs(save_dir, exist_ok=True)

    for file_name in listdir(patches_dir):
        slide_name = splitext(file_name)[0]
        makedirs(join(save_dir, slide_name))

        slide_path = join(slides_dir, f'{slide_name}.{slide_extension}')
        slide = open_slide(slide_path)
        print(f"{slide_path} loaded to have its patches be visualized.")

        try:
            objective_power = int(slide.properties['openslide.objective-power'])
        except KeyError:
            objective_power = 20

        if objective_power == 20:
            patch_size = 1024
            patch_level = 0 # downsample level at which to patch
        elif objective_power == 40:
            patch_size = 512
            patch_level = 1 # downsample level at which to patch

        with h5py.File(join(patches_dir, file_name), 'r') as f:
            coords = f['coords'][:]
            
        for coord1, coord2 in tqdm(coords):
            patch = slide.read_region((coord1, coord2), patch_level, (patch_size, patch_size)).convert("RGB")
            patch.save(join(save_dir, slide_name, f"patch_{coord1}_{coord2}.png"))