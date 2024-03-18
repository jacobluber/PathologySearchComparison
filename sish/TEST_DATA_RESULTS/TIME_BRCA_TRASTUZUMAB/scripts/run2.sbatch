#!/bin/bash

#SBATCH --job-name=run2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256gb
#SBATCH --gres=gpu:a100:2
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/sish/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/scripts/run2.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/sish/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/scripts/run2.out

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

cd /home/mxn2498/projects/new_search_comp/sish

# create_patches_fp
start_time=$(date +%s)
echo "Starting create_patches_fp.py at $(timestamp)"
python create_patches_fp.py --source /home/data/nejm_ai/TEST_DATA/WSI/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x/ --step_size 1024 --patch_size 1024 --seg --patch --stitch --patch_level 0 --save_dir ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp2/PATCHES/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x --preset BRCA_TRASTUZUMAB.csv || fail "python fail"
end_time=$(date +%s)
echo "create_patches.py finished at $(timestamp)"
print_duration $start_time $end_time

# extract_mosaic
echo "--------------------------------"
start_time=$(date +%s)
echo "Starting extract_mosaic.py at $(timestamp)"
python extract_mosaic.py --slide_data_path /home/data/nejm_ai/TEST_DATA/WSI/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x/  --slide_patch_path ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp2/PATCHES/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x/patches/ --save_path ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp2/MOSAICS/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x/ || fail "python fail"
end_time=$(date +%s)
echo "extract_mosaic.py finished at $(timestamp)"
print_duration $start_time $end_time

# artifacts_removal
echo "--------------------------------"
start_time=$(date +%s)
echo "Starting artifacts_removal.py at $(timestamp)"
python artifacts_removal.py --site_slide_path /home/data/nejm_ai/TEST_DATA/WSI/BRCA_TRASTUZUMAB/  --site_mosaic_path  ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp2/MOSAICS/BRCA_TRASTUZUMAB
end_time=$(date +%s)
echo "artifacts_removal.py finished at $(timestamp)"
print_duration $start_time $end_time

# build_index
echo "--------------------------------"
start_time=$(date +%s)
echo "Starting build_index.py at $(timestamp)"
python build_index.py --slide_path /home/data/nejm_ai/TEST_DATA/WSI --mosaic_path ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp2/MOSAICS/ --site BRCA_TRASTUZUMAB --slide_ext svs
end_time=$(date +%s)
echo "build_index.py finished at $(timestamp)"
print_duration $start_time $end_time

# overall
echo "--------------------------------"
overall_end=$(date +%s)
echo "Overall start time: $(timestamp -d @$overall_start)"
echo "Overall end time: $(timestamp -d @$overall_end)"
print_duration $overall_start $overall_end
