from os import makedirs
from os.path import join, basename, splitext, abspath
from statistics import mode, mean
from collections import Counter
import argparse
import random
import glob
import json
import pickle
import datetime
import sys
import json
import yaml

import numpy as np
from numpy.linalg import norm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import openslide
from matplotlib import pyplot as plt
import h5py
from PIL import Image
from tqdm import tqdm

from model import ccl_model


diagnoses_dict = {
    "Brain Lower Grade Glioma": "LGG",
    "Glioblastoma Multiforme": "GBM",
    "Breast Invasive Carcinoma": "BRCA",
    "Lung Adenocarcinoma": "LUAD",
    "Lung Squamous Cell Carcinoma": "LUSC",
    "Colon Adenocarcinoma": "COAD",
    "Liver Hepatocellular Carcinoma": "LIHC",
    "Cholangiocarcinoma": "CHOL",
}

sites_dict = {
    "Brain": "brain",
    "Breast": "breast",
    "Bronchus and lung": "lung",
    "Colon": "colon",
    "Liver and intrahepatic bile ducts": "liver",
}

sites_diagnoses_dict = {
    "brain": ["LGG", "GBM"],
    "breast": ["BRCA"],
    "lung": ["LUAD", "LUSC"],
    "colon": ["COAD"],
    "liver": ["LIHC", "CHOL"],
}



def cosine_sim(a, b):
    return np.dot(a, b)/(norm(a) * norm(b))


def calculate_weights(site):
    if site == "organ":
        factor = 30
        # Count the number of slide in each diagnosis (organ)
        latent_all = join(PATCHES_DIR, "*", "*", "patches", "*")
        type_of_organ = list(sites_diagnoses_dict.keys())
        total_slide = {k: 0 for k in type_of_organ}
        for latent_path in glob.glob(latent_all):
            anatomic_site = latent_path.split("/")[-4]
            total_slide[anatomic_site] += 1
    else:
        factor = 10
        # Count the number of slide in each site (organ)
        latent_all = join(PATCHES_DIR, site, "*", "patches", "*")
        type_of_diagnosis = sites_diagnoses_dict[site]
        total_slide = {k: 0 for k in type_of_diagnosis}
        for latent_path in glob.glob(latent_all):
            diagnosis = latent_path.split("/")[-3]
            total_slide[diagnosis] += 1
    
    # Using the inverse count as a weight for each diagnosis
    sum_inv = 0
    for v in total_slide.values():
        sum_inv += (1./v)

    # Set a parameter k  to make the weight sum to k (k = 10, here)
    norm_fact = factor / sum_inv
    weight = {k: norm_fact * 1./v for k, v in total_slide.items()}
    return weight
    

def wsi_query(mosaics, test_mosaics, metadata, site, weight, cosine_threshold, results_dir, temp_results_dir):
    Bags = {}    # Dictionary to store each Bag for each query WSI
    Entropies = {}    # Dictionary to store entropies for each patch in each Bag for each query WSI
    Etas = {}    # Dictionary to store eta thresholds for each patch in each Bag for each query WSI
    Results = {}    # Dictionary to store top-N similar WSIs to query WSI
    for fname in tqdm(test_mosaics.index.unique()):
        WSI = test_mosaics.loc[fname]["features"].tolist()
        k = len(WSI)
        Bag = {}
        Entropy = {}
        for patch_idx, patch_feature in enumerate(WSI):
            # Retreiving similar patches (creating Bag)
            if site == "organ":
                bag = [(idx, cosine_sim(patch_feature, row["features"])) for idx, row in mosaics.iterrows() if cosine_sim(patch_feature, row["features"]) >= cosine_threshold]
            else:
                site_mosaics = mosaics.loc[list(metadata.loc[mosaics.loc[:, "file_name"], "primary_site"].apply(lambda x: sites_dict[x]) == site)].copy()
                bag = [(idx, cosine_sim(patch_feature, row["features"])) for idx, row in site_mosaics.iterrows() if cosine_sim(patch_feature, row["features"]) >= cosine_threshold]
            Bag[patch_idx] = sorted(bag, key=lambda x: x[1], reverse=True)
            t = len(Bag[patch_idx])

            # Calculating entropy for each query patch in the Bag
            entropy = 0
            if site == "organ":
                u = set([sites_dict[metadata.loc[mosaics.loc[idx, "file_name"], "primary_site"]] for (idx, _) in Bag[patch_idx]])
                for organ in u:
                    num, denum = 0, 0
                    for (idx, sim) in Bag[patch_idx]:
                        bag_organ = sites_dict[metadata.loc[mosaics.loc[idx, "file_name"], "primary_site"]]
                        num += ((organ==bag_organ) * 1) * ((sim + 1) / 2) * weight[bag_organ]
                        denum += ((sim + 1) / 2) * weight[bag_organ]
                    p = num / denum
                    entropy -= p * np.log(p)
            else:
                u = set([diagnoses_dict[metadata.loc[site_mosaics.loc[idx, "file_name"], "project_name"]] for (idx, _) in Bag[patch_idx]])
                for diagnosis in u:
                    num, denum = 0, 0
                    for (idx, sim) in Bag[patch_idx]:
                        bag_diagnosis = diagnoses_dict[metadata.loc[site_mosaics.loc[idx, "file_name"], "project_name"]]
                        num += ((diagnosis==bag_diagnosis) * 1) * ((sim + 1) / 2) * weight[bag_diagnosis]
                        denum += ((sim + 1) / 2) * weight[bag_diagnosis]
                    p = num / denum
                    entropy -= p * np.log(p)
            Entropy[patch_idx] = entropy
            
        # Sorting Bag members in terms of descending entropy
        Bag = dict(sorted(Bag.items(), key=lambda x: Entropy[x[0]], reverse=True))

        # Calculating eta threshold for each query patch in the Bag
        eta_threshold = 0
        for patch_idx in range(len(WSI)):
            eta = np.mean([x[1] for x in Bag[patch_idx][:5]]) if len(Bag[patch_idx]) else 0
            # eta = 0 if np.isnan(eta) else eta
            eta_threshold += eta 
        eta_threshold = eta_threshold / k

        # Removing query patches in the Bag with small eta (eta < eta_threshold) 
        ids = []
        for idx, bag in Bag.items():
            eta = np.mean([x[1] for x in bag[:5]]) if len(bag) else 0
            # eta = 0 if np.isnan(eta) else eta
            if eta < eta_threshold:
                ids.append(idx)
        for idx in ids:
            del Bag[idx]

        # Majority voting for retrieving the results
        WSIRet = {}
        for idx, bag in Bag.items():
            if site == "organ":
                matches = [sites_dict[metadata.loc[mosaics.loc[b[0], "file_name"], "primary_site"]] for b in bag[:5]]
                slides = [mosaics.loc[b[0], "slide_path"] for b in bag[:5]]
            else:
                matches = [diagnoses_dict[metadata.loc[site_mosaics.loc[b[0], "file_name"], "project_name"]] for b in bag[:5]]
                slides = [site_mosaics.loc[b[0], "slide_path"] for b in bag[:5]]
            sims = [b[1] for b in bag[:5]]
            # Using slide path as the key
            slide_path = slides[matches.index(mode(matches))]
            if slide_path not in WSIRet:
                WSIRet[slide_path] = (slide_path, sims[matches.index(mode(matches))], mean(sims))
        WSIRet = list(WSIRet.values())

        with open(join(temp_results_dir, f"{splitext(fname)[0]}_bag.pkl"), "wb") as f:
            pickle.dump(Bag, f)
        with open(join(temp_results_dir, f"{splitext(fname)[0]}_entropy.pkl"), "wb") as f:
            pickle.dump(Entropy, f)
        with open(join(temp_results_dir, f"{splitext(fname)[0]}_WSIRet.pkl"), "wb") as f:
            pickle.dump(WSIRet, f)

        Bags[fname] = Bag
        Entropies[fname] = Entropy
        Etas[fname] = eta
        Results[fname] = WSIRet

    with open(join(results_dir, f"Bags.pkl"), "wb") as f:
        pickle.dump(Bags, f)
    with open(join(results_dir, f"Entropies.pkl"), "wb") as f:
        pickle.dump(Entropies, f)
    with open(join(results_dir, f"Etas.pkl"), "wb") as f:
        pickle.dump(Etas, f)
    with open(join(results_dir, f"Results.pkl"), "wb") as f:
        pickle.dump(Results, f)
        
    return Results, Bags, Entropies, Etas


def get_features(slide, patch):
    class roi_dataset(Dataset):
        def __init__(self, slide, patch, transforms):
            super().__init__()
            self.patch = patch
            self.slide = slide
            self.transforms = transforms

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            path = join(TEST_PATCHES_DIR, splitext(self.slide)[0], self.patch)
            patch_region = Image.open(path)            
            patch_region = self.transforms(patch_region)
            return patch_region
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    trnsforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dataset = roi_dataset(slide, patch, trnsforms)
    database_loader = DataLoader(dataset, batch_size=1024, shuffle=False)

    ccl = ccl_model(checkpoint_path="../../checkpoints/best_ckpt.pth").to(device)
    ccl.eval()
    
    with torch.no_grad():
        for batch in database_loader:
            batch = batch.to(device)
            features = ccl(batch)

    return features.cpu().numpy()
    

def patch_query(test_slide, patch, mosaics, metadata, site, cosine_threshold):
    _, coord1, coord2 = splitext(patch)[0].split("_")
    patch_feature = get_features(test_slide, patch)
    
    site_mosaics = mosaics.loc[list(metadata.loc[mosaics.loc[:, "file_name"], "primary_site"].apply(lambda x: sites_dict[x]) == site)].copy()
    
    bag = [(idx, cosine_sim(patch_feature, row["features"]), row["patch_level"], row["patch_size"], row["coord1"], row["coord2"]) for idx, row in site_mosaics.iterrows() if cosine_sim(patch_feature, row["features"]) >= cosine_threshold]
    Bag = sorted(bag, key=lambda x: x[1], reverse=True)
    
    slides = [site_mosaics.loc[b[0], "slide_path"] for b in Bag]
    meta = [b[1:] for b in Bag]
    Results = [(slide, sim, level, patch_size, coord1, coord2) for slide, (sim, level, patch_size, coord1, coord2) in zip(slides, meta)]
    return Results, Bag


parser = argparse.ArgumentParser(description='Searching through indexed dataset and queries.')
parser.add_argument('--experiment', type=str, required=True,
                    help='either UCLA, READER_STUDY, GBM_MICROSCOPE_CPTAC, GBM_MICROSCOPE_UPENN, BRCA_HER2, BRCA_TRASTUZUMAB.')
parser.add_argument('--cosine_threshold', type=float, default=0.7,
                    help='cosine threshold for RETCCL algorithm.')
parser.add_argument('--j', type=int, default=1)
    

if __name__ == "__main__":
    args = parser.parse_args()
    experiment = args.experiment
    cosine_threshold = args.cosine_threshold

    VIEW_URL = "https://portal.gdc.cancer.gov/files/"
    BASE = "/home/mxn2498/projects/new_search_comp/retccl/FEATURES"

    PATCHES_DIR = join(BASE, "DATABASE")
    TEST_SLIDES_DIR = join("/home/data/nejm_ai/TEST_DATA", experiment)
    metadata_path = join(PATCHES_DIR, "sampled_metadata.csv")
    mosaics_path = join(PATCHES_DIR, "mosaics.h5")
    test_mosaics_path = join(BASE, "TEST_DATA", experiment, "mosaics.json")
    RESULTS_DIR = f"/home/mxn2498/projects/new_search_comp/retccl/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/results{args.j}"

    TEST_PATCHES_DIR = join(BASE, "TEST_DATA", experiment, "visualized_patches")

    query_slides_path = join(BASE, "TEST_DATA", experiment, "query_slides.yaml")
    with open(query_slides_path, 'r') as f:
        query_slides = yaml.safe_load(f)

    metadata = pd.read_csv(metadata_path)
    metadata = metadata.set_index('file_name')

    mosaics = pd.read_hdf(mosaics_path, 'df')
    
    test_mosaics = pd.read_json(test_mosaics_path)
    test_mosaics["features"] = test_mosaics["features"].apply(lambda lst: torch.tensor(lst))
    test_mosaics["file_name"] = test_mosaics.apply(lambda row: basename(row["slide_path"]), axis=1)
    test_mosaics = test_mosaics.set_index(['file_name'], inplace=False)

    results_dir = join(RESULTS_DIR, "organ")
    temp_results_dir = join(results_dir, "temp")
    makedirs(temp_results_dir, exist_ok=True)

    weight = calculate_weights("organ")
    Results, Bags, Entropies, Etas = wsi_query(mosaics, test_mosaics, metadata, "organ", weight, cosine_threshold, results_dir, temp_results_dir)
