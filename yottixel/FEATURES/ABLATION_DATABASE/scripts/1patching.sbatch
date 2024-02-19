#!/bin/bash

#SBATCH --job-name=1patching
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --gres=gpu:a100:0
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/yottixel/FEATURES/DATABASE/scripts/1patching.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/yottixel/FEATURES/DATABASE/scripts/1patching.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/yottixel

python parallel_patching.py --data_dir /raid/nejm_ai/DATABASE/ --metadata_path ./FEATURES/DATABASE/sampled_metadata.csv --experiment DATABASE --save_dir ./FEATURES/DATABASE/PATCHES --num_processes 16 || fail "python fail"