# Running SISH
Before running the codes, make sure that you first change the directory to this directory:
```bash
cd sish
```
To run the SISH framework, we must first create the database index, then calculate test data features, and finally, carry out the search.

## Creating Database
The source code for this method requires the data to be stored in the following format:
```bash
DATA
└── WSI
    ├── SITE
    │   ├── DIAGNOSIS
    │   │   ├── RESOLUTION
    │   │   │   ├── slide_1
    │   │   │   ├── slide_2
    │   │   │   ├── slide_3
```
However, the data in `PathologySearchComparison/DATA/DATABASE/` is already stored like this:
```bash
DATA
└── DATABASE
    ├── PRIMARY_SITE
    │   ├── DIAGNOSIS
    │   │   ├── slide_1
    │   │   ├── slide_2
    │   │   ├── slide_3
```
To rearrange the data into the required format, you can run `rearrange_data_directory.py` for each diagnosis individually. For example:
```bash
python rearrange_data_directory.py --source_directory PathologySearchComparison/DATA/DATABASE/brain/GBM
```
This will create divide the slides into the following directories:
```bash
PathologySearchComparison/DATA/DATABASE/brain/GBM/20x
PathologySearchComparison/DATA/DATABASE/brain/GBM/40x
```
Please be advised that id you want to use the other methods with the same data, you have to manually revert this process. Once all slides are divided, we can start patching them using the following code for each resolution:
```bash
python create_patches_fp.py --source PathologySearchComparison/DATA/DATABASE/brain/GBM/20x/ --step_size 1024 --patch_size 1024 --seg --patch --stitch --save_dir ./FEATURES/DATABASE/PATCHES/brain/LGG/20x --preset tcga.csv
python create_patches_fp.py --source PathologySearchComparison/DATA/DATABASE/brain/GBM/40x/ --step_size 2048 --patch_size 2048 --seg --patch --stitch --save_dir ./FEATURES/DATABASE/PATCHES/brain/LGG/40x --preset tcga.csv
```
This will create `patches`, `masks`, and `stitches` subdirectories for each resolution. The next step is to  create mosaics from the resulting patches:
```bash
python extract_mosaic.py --slide_data_path PathologySearchComparison/DATA/DATABASE/brain/GBM/20x/ --slide_patch_path ./FEATURES/DATABASE/PATCHES/brain/LGG/20x/patches/ --save_path ./FEATURES/DATABASE/MOSAICS/brain/LGG/20x/
python extract_mosaic.py --slide_data_path PathologySearchComparison/DATA/DATABASE/brain/GBM/20x/ --slide_patch_path ./FEATURES/DATABASE/PATCHES/brain/LGG/40x/patches/ --save_path ./FEATURES/DATABASE/MOSAICS/brain/LGG/40x/
```
To remove unwanted artifacts from the results, you can run the following code:
```bash
python artifacts_removal.py --site_slide_path PathologySearchComparison/DATA/DATABASE/brain  --site_mosaic_path ./FEATURES/DATABASE/MOSAICS/brain
```
To build the index for the site retrieval task, you can run the following code:
```bash
python build_index_organ.py --site organ --mosaic_path ./FEATURES/DATABASE/MOSAICS/ --slide_path PathologySearchComparison/DATA/DATABASE --slide_ext .svs --checkpoint ./checkpoints/model_9.pt --codebook_semantic ./checkpoints/codebook_semantic.pt
```
And to build the index for sub-type retrieval task, for each diagnosis, you have to run `build_index.py` separately.
```bash
python build_index.py --site brain --mosaic_path ./FEATURES/DATABASE/MOSAICS/ --slide_path PathologySearchComparison/DATA/DATABASE --slide_ext .svs --checkpoint ./checkpoints/model_9.pt --codebook_semantic ./checkpoints/codebook_semantic.pt
```
And for the patch retrieval (for Reader study), you can run the following code. You need to use the `.csv` file located in `./FEATURES/PATCH_DATABASE`. Please be advised that the majority of these source codes have some modification compared to the original code published by their respective authors.
```bash
python build_index_patch.py --exp_name READER_STUDY --patch_label_file ./FEATURES/PATCH_DATABASE/summary.csv --patch_data_path ./FEATURES/PATCH_DATABASE/ALL
```
