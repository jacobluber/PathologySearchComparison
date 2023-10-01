#!/bin/bash

#SBATCH --job-name=1CPTAC
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
#SBATCH --error=/home/mxn2498/projects/new_search_comp/retccl/FEATURES/TEST_DATA/GBM_MICROSCOPE_CPTAC/scripts/1CPTAC.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/retccl/FEATURES/TEST_DATA/GBM_MICROSCOPE_CPTAC/scripts/1CPTAC.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/retccl

python create_patches_fp.py --source /raid/nejm_ai/TEST_DATA/GBM_MICROSCOPE_CPTAC/ --seg --patch --stitch --save_dir ./FEATURES/TEST_DATA/GBM_MICROSCOPE_CPTAC --preset GBM_MICROSCOPE_CPTAC.csv || fail "python fail"
python all_patches_visualization.py --slides_dir /raid/nejm_ai/TEST_DATA/GBM_MICROSCOPE_CPTAC --patches_dir ./FEATURES/TEST_DATA/GBM_MICROSCOPE_CPTAC/patches --save_dir ./FEATURES/TEST_DATA/GBM_MICROSCOPE_CPTAC/visualized_patches --slide_extension svs || fail "python fail"

echo "Job done successfully!"