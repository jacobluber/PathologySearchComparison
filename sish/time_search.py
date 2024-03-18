import argparse
import time
import os
from os import makedirs
from os.path import join, basename, splitext
import pickle
import glob
import operator
import copy
import math
from collections import Counter, defaultdict
import requests
import json
import io
import datetime
import sys
import yaml

import h5py
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import openslide
from matplotlib import pyplot as plt

from database import HistoDatabase

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


def unpickle_object(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    

def get_link(name):
    fields = ["file_name"]
    fields = ",".join(fields)

    # We want all svs files from 'Breas' primary site.
    filters = {
        "op": "in",
        "content":{
            "field": "file_name",
            "value": [name + ".svs"]
            }
    }

    params = {
        "filters": json.dumps(filters),
        "fields": fields,
        "format": "CSV",
        "size": "1"
    }

    response = requests.post(FILES_ENDPOINT, headers={"Content-Type": "application/json"}, json=params)
    df = pd.read_csv(io.StringIO(response.content.decode('utf-8')), dtype='object')
    
    return VIEW_URL + df.id[0]


def Uncertainty_Cal(bag, weight, is_organ=False):
    """
    Implementation of Weighted-Uncertainty-Cal in the paper.
    Input:
        bag (list): A list of dictionary which contain the searhc results for each mosaic
    Output:
        ent (float): The entropy of the mosaic retrieval results
        label_count (dict): The diagnois and the corresponding weight for each mosaic
        hamming_dist (list): A list of hamming distance between the input mosaic and the result
    """
    if len(bag) >= 1:
        label = []
        hamming_dist = []
        label_count = defaultdict(float)
        for bres in bag:
            if is_organ:
                label.append(bres['site'])
            else:
                label.append(bres['diagnosis'])
            hamming_dist.append(bres['hamming_dist'])

        # Counting the diagnoiss by weigted count
        # If the count is less than 1, round to 1
        for lb_idx, lb in enumerate(label):
            label_count[lb] += (1. / (lb_idx + 1)) * weight[lb]
        for k, v in label_count.items():
            if v < 1.0:
                v = 1.0
            else:
                label_count[k] = v

        # Normalizing the count to [0,1] for entropy calculation
        total = 0
        ent = 0
        for v in label_count.values():
            total += v
        for k in label_count.keys():
            label_count[k] = label_count[k] / total
        for v in label_count.values():
            ent += (-v * np.log2(v))
        return ent, label_count, hamming_dist
    else:
        return None, None, None
    

def Clean(len_info, bag_summary):
    """
    Implementation of Clean in the paper
    Input:
        len_info (list): The length of retrieval results for each mosaic
        bag_summary (list): A list that contains the positional index of mosaic,
        entropy, the hamming distance list, and the length of retrieval results
    Output:
        bag_summary (list): The same format as input one but without low quality result
        (i.e, result with large hamming distance)
        top5_hamming_distance (float): The mean of average hamming distance in top 5
        retrival results of all mosaics
    """
    LOW_FREQ_THRSH = 3
    LOW_PRECENT_THRSH = 5
    HIGH_PERCENT_THRSH = 95
    len_info = [b[-1] for b in bag_summary]
    if len(set(len_info)) <= LOW_FREQ_THRSH:
        pass
    else:
        bag_summary = [b for b in bag_summary if b[-1]
                       > np.percentile(len_info, LOW_PRECENT_THRSH)
                       and b[-1] < np.percentile(len_info, HIGH_PERCENT_THRSH)]

    # Remove the mosaic if its top5 mean hammign distance is bigger than average
    top5_hamming_dist = np.mean([np.mean(b[2][0:5]) for b in bag_summary])

    bag_summary = sorted(bag_summary, key=lambda x: (x[1]))  # sort by certainty
    bag_summary = [b for b in bag_summary if np.mean(b[2][0:5]) <= top5_hamming_dist]
    return bag_summary, top5_hamming_dist


def Filtered_BY_Prediction(bag_summary, label_count_summary):
    """
    Implementation of Filtered_By_Prediction in the paper
    Input:
        bag_summary (list): The same as the output from Clean
        label_count_summary (dict): The dictionary storing the diagnosis occurrence 
        of the retrieval result in each mosaic
    Output:
        bag_removed: The index (positional) of moaic that should not be considered 
        among the top5
    """
    voting_board = defaultdict(float)
    for b in bag_summary[0:5]:
        bag_index = b[0]
        for k, v in label_count_summary[bag_index].items():
            voting_board[k] += v
    final_vote_candidates = sorted(voting_board.items(), key=lambda x: -x[1])
    fv_pointer = 0
    while True:
        final_vote = final_vote_candidates[fv_pointer][0]
        bag_removed = {}
        for b in bag_summary[0:5]:
            bag_index = b[0]
            max_vote = max(label_count_summary[bag_index].items(), key=operator.itemgetter(1))[0]
            if max_vote != final_vote:
                bag_removed[bag_index] = 1
        if len(bag_removed) != len(bag_summary[0:5]):
            break
        else:
            fv_pointer += 1
    return bag_removed

def calculate_weights(site):
    if site == "organ":
        factor = 30
        # Count the number of slide in each diagnosis (organ)
        latent_all = join(DATA_DIR, "PATCHES", "*", "*", "*", "patches", "*")
        type_of_organ = [basename(e) for e in glob.glob(join(DATA_DIR, "PATCHES", "*"))]
        total_slide = {k: 0 for k in type_of_organ}
        for latent_path in glob.glob(latent_all):
            anatomic_site = latent_path.split("/")[-5]
            total_slide[anatomic_site] += 1
    else:
        factor = 10
        # Count the number of slide in each site (organ)
        latent_all = join(DATA_DIR, "PATCHES", site, "*", "*", "patches", "*")
        type_of_diagnosis = [basename(e) for e in glob.glob(join(DATA_DIR, "PATCHES", site, "*"))]
        total_slide = {k: 0 for k in type_of_diagnosis}
        for latent_path in glob.glob(latent_all):
            diagnosis = latent_path.split("/")[-4]
            total_slide[diagnosis] += 1
    
    # Using the inverse count as a weight for each diagnosis
    sum_inv = 0
    for v in total_slide.values():
        sum_inv += (1./v)

    # Set a parameter k  to make the weight sum to k (k = 10, here)
    norm_fact = factor / sum_inv
    weight = {k: norm_fact * 1./v for k, v in total_slide.items()}
    return weight


def save_results(query_slides, extension, site, experiment, j, codebook_semantic="checkpoints/codebook_semantic.pt",
          pre_step=375, succ_step=375, C=50, T=10, thrsh=128):
    
    save_dir = join(f"/home/mxn2498/projects/new_search_comp/sish/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp{j}", "Results", site)
    makedirs(save_dir, exist_ok=True)
    
    queries_latent_all = join(TEST_DATA_DIR, experiment, "LATENT", "*", "*", "*", "vqvae", "*")
    
    database_index_path = join(DATA_DIR, "DATABASES", site, "index_tree", "veb.pkl")
    index_meta_path = join(DATA_DIR, "DATABASES", site, "index_meta", "meta.pkl")
    db = HistoDatabase(database_index_path, index_meta_path, codebook_semantic)
    
    results = {}
    for latent_path in glob.glob(queries_latent_all):
        resolution = latent_path.split("/")[-3]
        diagnosis = latent_path.split("/")[-4]
        anatomic_site = latent_path.split("/")[-5]
        slide_id = basename(latent_path).replace(".h5", "")

        densefeat_path = latent_path.replace("vqvae", "densenet").replace(".h5", ".pkl")
        slide_path = os.path.join(WSI_DIR, anatomic_site, diagnosis, f"{slide_id}.{extension}")

        db.leave_test_slides(list(query_slides.keys()))

        with h5py.File(latent_path, 'r') as hf:
            feat = hf['features'][:]
            coords = hf['coords'][:]
        with open(densefeat_path, 'rb') as handle:
            densefeat = pickle.load(handle)

        tmp_res = []
        for idx, patch_latent in enumerate(feat):
            res = db.query(patch_latent, densefeat[idx], pre_step, succ_step, C, T, thrsh)
            tmp_res.append(res)

        key = slide_id
        results[key] = {'results': None, 'label_query': None}
        results[key]['results'] = tmp_res
        if site == 'organ':
            results[key]['label_query'] = anatomic_site
        else:
            results[key]['label_query'] = diagnosis
    
    with open(join(save_dir, f"results.pkl"), 'wb') as handle:
        pickle.dump(results, handle)
    
    return results

def query(results, site, topK_mMV):
    query_results = []
    for test_slide in results.keys():
        test_slide_result = results[test_slide]['results']
        
        # Filter out complete failure case (i.e.,
        # All mosaics fail to retrieve a patch that meet the criteria)
        ttlen = 0
        for tt in test_slide_result:
            ttlen += len(tt)
        if ttlen == 0:
            continue

        bag_result = []
        bag_summary = []
        len_info = []
        label_count_summary = {}
        weight = calculate_weights(site)
        for idx, bag in enumerate(test_slide_result):
            if site == "organ":
                ent, label_cnt, dist = Uncertainty_Cal(bag, weight, is_organ=True)
            else:
                ent, label_cnt, dist = Uncertainty_Cal(bag, weight, is_organ=False)

            if ent is not None:
                label_count_summary[idx] = label_cnt
                bag_summary.append((idx, ent, dist, len(bag)))
                len_info.append(len(bag))

        bag_summary_dirty = copy.deepcopy(bag_summary)
        bag_summary, hamming_thrsh = Clean(len_info, bag_summary)
        bag_removed = Filtered_BY_Prediction(bag_summary, label_count_summary)

        # Process to calculate the final ret slide
        ret_final = []
        visited = {}
        for b in bag_summary:
            bag_index = b[0]
            uncertainty = b[1]
            res = results[test_slide]['results'][bag_index]
            for r in res:
                if uncertainty == 0:
                    if r['slide_name'] not in visited:
                        if site == "organ":
                            ret_final.append((r['slide_name'], r['hamming_dist'], r['site'], uncertainty, bag_index))
                        else:
                            ret_final.append((r['slide_name'], r['hamming_dist'], r['diagnosis'], uncertainty, bag_index))
                        visited[r['slide_name']] = 1
                else:
                    if (r['hamming_dist'] <= hamming_thrsh) and (r['slide_name'] not in visited):
                        if site == "organ":
                            ret_final.append((r['slide_name'], r['hamming_dist'], r['site'], uncertainty, bag_index))
                        else:
                            ret_final.append((r['slide_name'], r['hamming_dist'], r['diagnosis'], uncertainty, bag_index))
                        visited[r['slide_name']] = 1

        ret_final_tmp = [(e[1], e[2], e[3], e[-1]) for e in sorted(ret_final, key=lambda x: (x[3], x[1]))
                         if e[-1] not in bag_removed]
        ret_final = [(e[0], e[1], e[2]) for e in sorted(ret_final, key=lambda x: (x[3], x[1]))
                     if e[-1] not in bag_removed][0:topK_mMV]

        query_results.append((test_slide, ret_final))
    return query_results


parser = argparse.ArgumentParser(description='Searching through indexed dataset and queries.')
parser.add_argument('--experiment', type=str, required=True,
                    help='either UCLA, READER_STUDY, GBM_MICROSCOPE_CPTAC, GBM_MICROSCOPE_UPENN, BRCA_HER2, BRCA_TRASTUZUMAB.')
parser.add_argument('--extension', type=str, default='svs', help='ndpi for GBM_MICROSCOPE_UPENN, svs everywhere else.')
parser.add_argument('--j', type=int)


if __name__ == "__main__":
    args = parser.parse_args()
    experiment = args.experiment
    extension = args.extension

    VIEW_URL = "https://portal.gdc.cancer.gov/files/"
    FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

    DATA_DIR = "FEATURES/DATABASE/"
    TEST_DATA_DIR = "FEATURES/TEST_DATA/"

    WSI_DIR = "/raid/nejm_ai/DATABASE/"
    TEST_WSI_DIR = "/raid/nejm_ai/TEST_DATA/"
    
    pre_step = 375
    succ_step = 375
    C = 50
    T = 10
    thrsh = 128

    metadata = pd.read_csv("FEATURES/DATABASE/sampled_metadata.csv")
    metadata = metadata.set_index('file_name')

    with open(f"FEATURES/TEST_DATA/{experiment}/query_slides.yaml", 'r') as f:
        query_slides = yaml.safe_load(f)

    query_slides_proxies = dict()
    for key in query_slides.keys():
        query_slides_proxies[splitext(key)[0]] = splitext(key)[0]

    sites = ["organ"]
    for site in sites:
        print(f"processing for {site} started.")
        save_results(query_slides, extension=extension, site=site, experiment=experiment, j=args.j)
        