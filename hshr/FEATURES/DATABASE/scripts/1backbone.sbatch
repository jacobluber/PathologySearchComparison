#!/bin/bash

#SBATCH --job-name=1backbone
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128gb
#SBATCH --gres=gpu:0
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/scripts/1backbone.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/scripts/1backbone.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/hshr/backbone

python generate.py --SVS_DIR /raid/nejm_ai/DATABASE/ --RESULT_DIR ../FEATURES/DATABASE/BACKBONE/ --TMP ../FEATURES/DATABASE/TEMP_BACKBONE/ --SIZE 1000 || fail "python fail"