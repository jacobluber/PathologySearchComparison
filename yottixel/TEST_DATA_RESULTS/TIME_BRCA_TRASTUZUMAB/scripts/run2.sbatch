#!/bin/bash

#SBATCH --job-name=run2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256gb
#SBATCH --gres=gpu:a100:2
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/yottixel/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/scripts/run2.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/yottixel/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/scripts/run2.out

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

cd /home/mxn2498/projects/new_search_comp/yottixel

# parallel_patching
start_time=$(date +%s)
echo "Starting parallel_patching.py at $(timestamp)"
python parallel_patching.py --data_dir /home/data/nejm_ai/TEST_DATA/WSI/ --metadata_path ./FEATURES/TEST_DATA/BRCA_TRASTUZUMAB/test_metadata.csv --experiment BRCA_TRASTUZUMAB --save_dir ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp2/PATCHES --num_processes 8
end_time=$(date +%s)
echo "parallel_patching.py finished at $(timestamp)"
print_duration $start_time $end_time

# feature_extraction
echo "--------------------------------"
start_time=$(date +%s)
echo "Starting feature_extraction.py at $(timestamp)"
python feature_extraction.py --patch_dir ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp2/PATCHES/ --experiment BRCA_TRASTUZUMAB --network_input_patch_width 1000 --extracted_features_save_adr ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp2/features.pkl --batch_size 256 --use_gpu True
end_time=$(date +%s)
echo "feature_extraction.py finished at $(timestamp)"
print_duration $start_time $end_time

# overall
echo "--------------------------------"
overall_end=$(date +%s)
echo "Overall start time: $(timestamp -d @$overall_start)"
echo "Overall end time: $(timestamp -d @$overall_end)"
print_duration $overall_start $overall_end
