#!/bin/bash

#SBATCH --job-name=3h
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --gres=gpu:a100:1
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/scripts/logs/3h.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/scripts/logs/3h.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/hshr

# python preprocess.py --WSI_DIR /raid/nejm_ai/DATABASE/brain --SLIDE_EXT svs --RESULT_DIR /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/brain --TMP /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/TEMP || fail "python fail"
# python preprocess.py --WSI_DIR /raid/nejm_ai/DATABASE/breast --SLIDE_EXT svs --RESULT_DIR /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/breast --TMP /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/TEMP || fail "python fail"
python preprocess.py --WSI_DIR /raid/nejm_ai/DATABASE/colon --SLIDE_EXT svs --RESULT_DIR /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/colon --TMP /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/TEMP || fail "python fail"
# python preprocess.py --WSI_DIR /raid/nejm_ai/DATABASE/liver --SLIDE_EXT svs --RESULT_DIR /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/liver --TMP /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/TEMP || fail "python fail"
# python preprocess.py --WSI_DIR /raid/nejm_ai/DATABASE/lung --SLIDE_EXT svs --RESULT_DIR /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/lung --TMP /home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/TEMP || fail "python fail"

echo "Job done successfully!"