import argparse
import pickle
from os import makedirs
from os.path import join, abspath

import h5py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Generating mosaics from extracted features..")
parser.add_argument('--features_path', default='./FEATURES/DATABASE/features.h5', help='path to features.h5')
parser.add_argument('--save_dir', default='./FEATURES/DATABASE', help='directory to save temporary logs and final results.')
parser.add_argument('--kl', type=int, default=9 , help='number of clusters.')
parser.add_argument('--R', type=float, default=0.2, help='ratio of patches to be added to mosaic from each cluster..')


if __name__ == "__main__":
    args = parser.parse_args()
    features_path = abspath(args.features_path)
    save_dir = abspath(args.save_dir)
    kl = args.kl
    R = args.R

    features_df = pd.read_hdf(features_path, 'df')
    features_df = features_df.set_index(['file_name'], inplace=False)
    
    j = 0
    mosaics = pd.DataFrame(columns=features_df.columns.append(pd.Index(["feature_cluster"])))
    for slide_id in tqdm(features_df.index.unique()):
        print(f"started mosaic generation for {slide_id}...")
        kmeans = KMeans(n_clusters=kl, n_init=10, random_state=0)
        features = np.stack(features_df.loc[slide_id, "features"].values)
        kmeans.fit(features)
        features_df.loc[slide_id, 'feature_cluster'] = kmeans.labels_

        slide_df = features_df.loc[slide_id].copy()
        mosaic = pd.DataFrame(columns=slide_df.columns)
        for i in range(kl):
            cluster_patches = slide_df[slide_df.loc[slide_id, "feature_cluster"]==i]
            n_selected = max(1, int(len(cluster_patches) * R))

            km = KMeans(n_clusters=n_selected, n_init=10, random_state=0)
            loc_features = [[row["coord1"], row["coord2"]] for _, row in cluster_patches.iterrows()]
            ds = km.fit_transform(loc_features)

            c_selected_idx = []
            for idx in range(n_selected):
                sorted_idx = np.argsort(ds[:, idx])
                for sidx in sorted_idx:
                    if sidx not in c_selected_idx:
                        c_selected_idx.append(sidx)
                        mosaic = pd.concat([mosaic, cluster_patches.iloc[sidx:sidx+1]], ignore_index=True)
                        break

        path = join(save_dir, "_mosaics_logs")
        makedirs(path, exist_ok=True)
        with open(join(path, f"mosaic{j}.pkl"), "wb") as f:
            pickle.dump(mosaic, f)
        j +=1

        mosaics = pd.concat([mosaics, mosaic], ignore_index=True)

    mosaics.to_hdf(join(save_dir, "mosaics.h5"), key="df", mode="w")
    features_df.to_hdf(join(save_dir, "features_with_cluster.h5"), key="df", mode="w")
    
    print("mosaic generation completed successfully.")
    