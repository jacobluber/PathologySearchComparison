#!/bin/bash
#SBATCH --job-name=gdc_download
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mohammadsadegh.nasr@mavs.uta.edu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb
#SBATCH --time=120:00:00
#SBATCH --error=/home/mxn2498/projects/PathologySearchComparison/data/DATABASE/gdc_download.err
#SBATCH --output=/home/mxn2498/projects/PathologySearchComparison/data/DATABASE/gdc_download.out

date; hostname; pwd

echo "Download Started!"

gdc-client download -m /home/mxn2498/projects/PathologySearchComparison/data/DATABASE/download_manifest.txt --config /home/mxn2498/projects/PathologySearchComparison/data/DATABASE/dtt-config.dtt

echo "Download Finished!"

date
