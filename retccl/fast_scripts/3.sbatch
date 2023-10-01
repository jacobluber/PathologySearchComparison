#!/bin/bash

#SBATCH --job-name=3
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --gres=gpu:0
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/retccl/fast_scripts/3.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/retccl/fast_scripts/3.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/retccl

# python search.py --experiment UCLA || fail "python fail"
# python search.py --experiment READER_STUDY || fail "python fail"
# python search.py --experiment GBM_MICROSCOPE_UPENN || fail "python fail"
# python search.py --experiment GBM_MICROSCOPE_CPTAC || fail "python fail"
python search.py --experiment BRCA_TRASTUZUMAB || fail "python fail"
# python search.py --experiment BRCA_HER2 || fail "python fail"

echo "Jobs done successfully!"