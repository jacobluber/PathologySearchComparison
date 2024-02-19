#!/bin/bash

#SBATCH --job-name=UPENN
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --partition=conference
#SBATCH --account=conference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/yottixel/FEATURES/TEST_DATA/GBM_MICROSCOPE_UPENN/scripts/UPENN.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/yottixel/FEATURES/TEST_DATA/GBM_MICROSCOPE_UPENN/scripts/UPENN.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/yottixel

python parallel_patching.py --data_dir /raid/nejm_ai/TEST_DATA/ --metadata_path ./FEATURES/TEST_DATA/GBM_MICROSCOPE_UPENN/test_metadata.csv --experiment GBM_MICROSCOPE_UPENN --save_dir ./FEATURES/TEST_DATA/GBM_MICROSCOPE_UPENN/PATCHES --num_processes 16 || fail "python fail"
python feature_extraction.py --patch_dir ./FEATURES/TEST_DATA/GBM_MICROSCOPE_UPENN/PATCHES/ --experiment GBM_MICROSCOPE_UPENN --network_input_patch_width 1000 --extracted_features_save_adr ./FEATURES/TEST_DATA/GBM_MICROSCOPE_UPENN/features.pkl --batch_size 256 --use_gpu True || fail "python fail"