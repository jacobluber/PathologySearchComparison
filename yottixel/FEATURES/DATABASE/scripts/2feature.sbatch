#!/bin/bash

#SBATCH --job-name=2feature
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --gres=gpu:a100:2
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/yottixel/FEATURES/DATABASE/scripts/2feature.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/yottixel/FEATURES/DATABASE/scripts/2feature.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"

cd /home/mxn2498/projects/new_search_comp/yottixel

python feature_extraction.py --patch_dir ./FEATURES/DATABASE/PATCHES/ --experiment DATABASE --extracted_features_save_adr ./FEATURES/DATABASE/features.pkl --batch_size 256 --use_gpu True || fail "python fail"