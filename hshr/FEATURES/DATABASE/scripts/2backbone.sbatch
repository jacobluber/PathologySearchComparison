#!/bin/bash

#SBATCH --job-name=2backbone
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --gres=gpu:2
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/scripts/2backbone.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/hshr/FEATURES/DATABASE/scripts/2backbone.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/hshr/backbone/SimCLR

python run.py --pathology_root ../../FEATURES/DATABASE/BACKBONE --save_path ../../FEATURES/DATABASE/BACKBONE_RESULTS/ || fail "python fail"