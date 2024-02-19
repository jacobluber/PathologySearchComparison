#!/bin/bash

#SBATCH --job-name=run4
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --gres=gpu:a100:1
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/retccl/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/scripts/run4.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/retccl/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/scripts/run4.out

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

function timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

function print_duration() {
    start_time=$1
    end_time=$2
    duration=$(date -u -d "0 $end_time seconds - $start_time seconds" +"%H:%M:%S")
    echo "Duration: $duration"
}

overall_start=$(date +%s)

source /home/mxn2498/miniconda3/etc/profile.d/conda.sh || fail "conda load fail"
conda activate search1 || fail "conda activate fail"

cd /home/mxn2498/projects/new_search_comp/retccl

# create_patches_fp
start_time=$(date +%s)
echo "Starting create_patches_fp.py at $(timestamp)"
python create_patches_fp.py --source /home/data/nejm_ai/TEST_DATA/BRCA_TRASTUZUMAB/ --seg --patch --stitch --save_dir ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp4/BRCA_TRASTUZUMAB --preset BRCA_TRASTUZUMAB.csv
end_time=$(date +%s)
echo "create_patches_fp.py finished at $(timestamp)"
print_duration $start_time $end_time

# extract_features
echo "--------------------------------"
start_time=$(date +%s)
echo "Starting extract_features.py at $(timestamp)"
python extract_features.py --patch_dataframe_path ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/patch_dataframe.csv --save_dir ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp4/BRCA_TRASTUZUMAB
end_time=$(date +%s)
echo "extract_features.py finished at $(timestamp)"
print_duration $start_time $end_time

# generate_mosaics
echo "--------------------------------"
start_time=$(date +%s)
echo "Starting generate_mosaics.py at $(timestamp)"
python generate_mosaics.py --features_path ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp4/BRCA_TRASTUZUMAB/features.h5 --save_dir ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp4/BRCA_TRASTUZUMAB --kl 9 --R 0.2
end_time=$(date +%s)
echo "generate_mosaics.py finished at $(timestamp)"
print_duration $start_time $end_time

# overall
echo "--------------------------------"
overall_end=$(date +%s)
echo "Overall start time: $(timestamp -d @$overall_start)"
echo "Overall end time: $(timestamp -d @$overall_end)"
print_duration $overall_start $overall_end
