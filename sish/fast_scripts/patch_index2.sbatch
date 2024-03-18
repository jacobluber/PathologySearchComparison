#!/bin/bash

#SBATCH --job-name=patch_index2
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
#SBATCH --error=/home/mxn2498/projects/search_comparision/sish_new_test/patch_index2.err
#SBATCH --output=/home/mxn2498/projects/search_comparision/sish_new_test/patch_index2.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/search_comparision/sish_new_test

python build_index_patch2.py --exp_name READER_STUDY --patch_label_file DATA_PATCH/summary.csv --patch_data_path DATA_PATCH/All || fail "python fail"
