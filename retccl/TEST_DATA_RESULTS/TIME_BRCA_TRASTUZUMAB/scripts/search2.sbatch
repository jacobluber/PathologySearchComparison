#!/bin/bash

#SBATCH --job-name=search2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128gb
#SBATCH --gres=gpu:a100:0
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/new_search_comp/retccl/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/scripts/search2.err
#SBATCH --output=/home/mxn2498/projects/new_search_comp/retccl/TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/scripts/search2.out

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

# time_search
start_time=$(date +%s)
echo "Starting time_search.py at $(timestamp)"
python time_search.py --experiment BRCA_TRASTUZUMAB --j 2 || fail "python fail"
end_time=$(date +%s)
echo "time_search.py finished at $(timestamp)"
print_duration $start_time $end_time

# overall
echo "--------------------------------"
overall_end=$(date +%s)
echo "Overall start time: $(timestamp -d @$overall_start)"
echo "Overall end time: $(timestamp -d @$overall_end)"
print_duration $overall_start $overall_end
