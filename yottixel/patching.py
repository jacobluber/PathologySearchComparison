from os import makedirs
from os.path import join, splitext
import argparse
import ast

import h5py
import openslide
import numpy as np
import pandas as pd
from cv2 import filter2D
from tqdm import tqdm
from sklearn.cluster import KMeans


def RGB2HSD(X):
    eps = np.finfo(float).eps
    X[np.where(X==0.0)] = eps
    
    OD = -np.log(X / 1.0)
    D  = np.mean(OD,3)
    D[np.where(D==0.0)] = eps
    
    cx = OD[:,:,:,0] / (D) - 1.0
    cy = (OD[:,:,:,1]-OD[:,:,:,2]) / (np.sqrt(3.0)*D)
    
    D = np.expand_dims(D,3)
    cx = np.expand_dims(cx,3)
    cy = np.expand_dims(cy,3)
            
    X_HSD = np.concatenate((D,cx,cy),3)
    return X_HSD

def clean_thumbnail(thumbnail):
    thumbnail_arr = np.asarray(thumbnail)
    # writable thumbnail
    wthumbnail = np.zeros_like(thumbnail_arr)
    wthumbnail[:, :, :] = thumbnail_arr[:, :, :]
    # Remove pen marking here
    # We are skipping this
    # This  section sets regoins with white spectrum as the backgroud regoin
    thumbnail_std = np.std(wthumbnail, axis=2)
    wthumbnail[thumbnail_std<5] = (np.ones((1,3), dtype="uint8")*255)
    thumbnail_HSD = RGB2HSD( np.array([wthumbnail.astype('float32')/255.]) )[0]
    kernel = np.ones((30,30),np.float32)/900
    thumbnail_HSD_mean = filter2D(thumbnail_HSD[:,:,2],-1,kernel)
    wthumbnail[thumbnail_HSD_mean<0.05] = (np.ones((1,3),dtype="uint8")*255)
    return wthumbnail

def save_mosaic(mosaic, patch_save_dir):
    df = pd.DataFrame(mosaic)
    df['loc'] = df['loc'].apply(lambda x: str(x))
    df['wsi_loc'] = df['wsi_loc'].apply(lambda x: str(x))
    df['rgb_histogram'] = df['rgb_histogram'].apply(lambda x: str(x.tolist()))  # Convert numpy array to list before converting to string
    df.to_hdf(join(patch_save_dir, 'mosaic.h5'), key='df', mode='w')

def load_mosaic(mosaic_path):
    df = pd.read_hdf(mosaic_path, 'df')
    # Convert back to original forms
    df['loc'] = df['loc'].apply(ast.literal_eval)
    df['wsi_loc'] = df['wsi_loc'].apply(ast.literal_eval)
    df['rgb_histogram'] = df['rgb_histogram'].apply(lambda x: np.array(ast.literal_eval(x)))
    return df.to_dict('records')


parser = argparse.ArgumentParser(description="segmenting and patching")
parser.add_argument("--data_dir", type=str, default="/home/data/GDC_BBCLL/", help="Address to the folder containing raw wsi image files.")
parser.add_argument("--metadata_path", type=str, default="./sampled_metadata.csv", help="Path to the metadata .csv file.")
parser.add_argument("--save_dir", type=str, default="./PATCHES", help="Directory to save patches.")


if __name__ == "__main__":
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata_path)
    data_dir = args.data_dir
    save_dir = args.save_dir
    makedirs(save_dir, exist_ok=True)
    
    for index, row in metadata.iterrows():
        patch_save_dir = join(save_dir, splitext(row["file_name"])[0])
        makedirs(patch_save_dir, exist_ok=True)
        
        tcga_slide = join(data_dir, row["id"], row["file_name"])
        slide = openslide.open_slide(tcga_slide)
        print(f"{patch_save_dir} loaded to be processed...")

        thumbnail = slide.get_thumbnail((500, 500))
        cthumbnail = clean_thumbnail(thumbnail)
        tissue_mask = (cthumbnail.mean(axis=2) != 255) * 1.0

        try:
            objective_power = int(slide.properties['openslide.objective-power'])
        except KeyError:
            objective_power = 20

        w, h = slide.dimensions
        # at 20x its 1000x1000
        # at 40x its 2000x2000
        patch_size = int((objective_power/20.)*1000)
        
        mask_hratio = (tissue_mask.shape[0]/h)*patch_size
        mask_wratio = (tissue_mask.shape[1]/w)*patch_size
        
        # iterating over patches
        patches = []
        for i, hi in enumerate(range(0, h, int(patch_size))):
            _patches = []
            for j, wi in enumerate(range(0, w, int(patch_size))):
                # check if patch contains 70% tissue area
                mi = int(i * mask_hratio)
                mj = int(j * mask_wratio)
                patch_mask = tissue_mask[mi:mi + int(mask_hratio), mj:mj + int(mask_wratio)]
                tissue_coverage = np.count_nonzero(patch_mask) / patch_mask.size
                _patches.append({'loc': [i, j], 'wsi_loc': [int(hi), int(wi)], 'tissue_coverage': tissue_coverage})
            patches.append(_patches)

        # for patch to be considered it should have this much tissue area
        tissue_threshold = 0.7

        #Next step in the pipeline is to calculate the RGB histogram for each patch (but at 5x).
        print(f"Histogram calculation started...")
        flat_patches = np.ravel(patches)
        for patch in tqdm(flat_patches):
            # ignore patches with less tissue coverage
            if patch['tissue_coverage'] < tissue_threshold:
                continue
            # this loc is at the objective power
            h, w = patch['wsi_loc']
            # we will go one level lower, i.e. (objective power / 4)
            # we still need patches at 5x of size 250x250
            # this logic can be modified and may not work properly for images of lower objective power < 20 or greater than 40
            patch_size_5x = int(((objective_power / 4) / 5) * 250.)
            patch_region = slide.read_region((w, h), 1, (patch_size_5x, patch_size_5x)).convert('RGB')
            if patch_region.size[0] != 250:
                patch_region = patch_region.resize((250, 250))
            histogram = (np.array(patch_region)/255.).reshape((250*250, 3)).mean(axis=0)
            patch['rgb_histogram'] = histogram

        # Now, run k-means on the RGB histogram features for all selected patches
        print(f"Tissue coverage filtering started...")
        selected_patches_flags = [patch['tissue_coverage'] >= tissue_threshold for patch in flat_patches]
        selected_patches = flat_patches[selected_patches_flags]

        print(f"RGB clustering started...")
        kmeans_clusters = 9
        kmeans = KMeans(n_clusters=kmeans_clusters, n_init=10, random_state=0)
        features = np.array([entry['rgb_histogram'] for entry in selected_patches])
        
        kmeans.fit(features)

        #Selecting Mosaic
        # Another hyperparameter of Yottixel
        # Yottixel has been tested with 5, 10, and 15 with 15 performing most optimally
        percentage_selected = 15

        print(f"Mosaic generation started...")
        mosaic = []
        for i in range(kmeans_clusters):
            cluster_patches = selected_patches[kmeans.labels_ == i]
            n_selected = max(1, int(len(cluster_patches)*percentage_selected/100.))
            km = KMeans(n_clusters=n_selected, n_init=10, random_state=0)
            loc_features = [patch['wsi_loc'] for patch in cluster_patches]
            ds = km.fit_transform(loc_features)
            c_selected_idx = []
            for idx in range(n_selected):
                sorted_idx = np.argsort(ds[:, idx])
                for sidx in sorted_idx:
                    if sidx not in c_selected_idx:
                        c_selected_idx.append(sidx)
                        mosaic.append(cluster_patches[sidx])
                        break
        save_mosaic(mosaic, patch_save_dir)
        print(f"{len(mosaic)} mosaics generated...")
        
        for patch in tqdm(mosaic):
            # this loc is at the objective power
            h, w = patch['wsi_loc']
            patch_size_20x = int((objective_power/20.)*1000)
            patch_region = slide.read_region((w, h), 0, (patch_size_20x, patch_size_20x)).convert('RGB')

            if objective_power == 40:
                new_size = (patch_size_20x // 2, patch_size_20x // 2)
                patch_region = patch_region.resize(new_size)

            # Save the patch_region as a JPG file
            output_file = join(patch_save_dir, f"patch_{patch['loc'][0]}_{patch['loc'][1]}.jpg") 
            patch_region.save(output_file, 'JPEG')
        
        print(f"All patches succesfully saved...\n")
        slide.close()
    