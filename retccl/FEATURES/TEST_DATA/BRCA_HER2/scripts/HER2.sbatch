#!/bin/bash

#SBATCH --job-name=HER2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --partition=conference
#SBATCH --account=conference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --gres=gpu:1
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/retccl/FEATURES/TEST_DATA/BRCA_HER2/scripts/HER2.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/retccl/FEATURES/TEST_DATA/BRCA_HER2/scripts/HER2.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/retccl

python extract_features.py --patch_dataframe_path ./FEATURES/TEST_DATA/BRCA_HER2/patch_dataframe.csv --save_dir ./FEATURES/TEST_DATA/BRCA_HER2 || fail "python fail"
python generate_mosaics.py --features_path ./FEATURES/TEST_DATA/BRCA_HER2/features.h5 --save_dir ./FEATURES/TEST_DATA/BRCA_HER2 --kl 9 --R 0.2 || fail "python fail"

echo "Job done successfully!"