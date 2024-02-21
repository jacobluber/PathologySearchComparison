#!/bin/bash

#SBATCH --job-name=2HSHR
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
#SBATCH --error=/home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/scripts/logs/2HSHR.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/scripts/logs/2HSHR.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/hshr

python ssl_encoder_training.py --RESULT_DIR /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE --TMP /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/TEMP --MODEL_DIR /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/model --DATASETS brain breast colon liver lung || fail "python fail"

echo "Job done successfully!"