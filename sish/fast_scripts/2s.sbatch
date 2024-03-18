#!/bin/bash

#SBATCH --job-name=2s
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --gres=gpu:0
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/sish/fast_scripts/2s.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/sish/fast_scripts/2s.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/sish

python search.py --experiment BRCA_TRASTUZUMAB --extension svs || fail "python fail"

echo "Jobs done successfully!"