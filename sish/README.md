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

## Test Data Featrue Extraction
Here, we would essentially follow the same procedure as above to patch and extract the features from test slides. First, we need to chage the format of directory for each test set. For the sake of simplicity, we can just duplicate the name of the dataset instead of having `SITE\DIAGNOSIS`. For examle, for the Yale HER2 experiment, these are the commands that need be run:
```bash
python create_patches_fp.py --source PathologySearchComparison/DATA/TEST_DATA/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x/ --step_size 1024 --patch_size 1024 --seg --patch --stitch --patch_level 0 --save_dir ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp1/PATCHES/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x --preset BRCA_TRASTUZUMAB.csv
python extract_mosaic.py --slide_data_path PathologySearchComparison/DATA/TEST_DATA/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x/  --slide_patch_path ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp1/PATCHES/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x/patches/ --save_path ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp1/MOSAICS/BRCA_TRASTUZUMAB/BRCA_TRASTUZUMAB/20x/
python artifacts_removal.py --site_slide_path PathologySearchComparison/DATA/TEST_DATA/BRCA_TRASTUZUMAB/  --site_mosaic_path  ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp1/MOSAICS/BRCA_TRASTUZUMAB
python build_index.py --slide_path PathologySearchComparison/DATA/TEST_DATA/ --mosaic_path ./TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/temp1/MOSAICS/ --site BRCA_TRASTUZUMAB --slide_ext svs
```
It worths mentioning the `slide2.svs` from the UCLA study was very large for this code to handle. We made some changes to the source code to divide it into 4 section, patch them individually, and then add all regions together. The first part is done automatically in the source code, but for adding everything together, you can use the `test_post_process.ipynb` notebook.

Except for the Reader study, other experiments do not study patch level retrieval. And for the patch retrieval (for Reader study), you can run the following code. You need to use the `summary.csv` file located in `./FEATURES/PATCH_DATABASE`. Please be advised that the majority of these source codes have some modification compared to the original code published by their respective authors.
```bash
python build_index_patch.py --exp_name READER_STUDY --patch_label_file ./FEATURES/PATCH_DATABASE/summary.csv --patch_data_path ./FEATURES/PATCH_DATABASE/ALL
```

## Searching the Database for Test Slides
Once you extract the features as outlined above, you can use the `search_template.ipynb` notebook located at `TEST_DATA_RESULTS` to carry out the search and store the performance metrics. For our experiments, we have provided the notebook we have used to store the data in each corresponding subdirectory (e.g. `TEST_DATA_RESULTS/ABLATION_BRCA_HER2/search.ipynb`).

## Runtime Analysis
We have also provided the code we have used to evaluate the runtime of the model for the Yale Trastuzumab study under `TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/` directory. In the `scripts` directory, you can find the logs for patching and feature extraction, and in the accompanied notebook, you can find the search and retrieval runtimes.