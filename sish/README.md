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





python build_index_patch.py --exp_name patch_search_test --patch_label_file ./DATA_PATCH_TEST/summary.csv --patch_data_path ./DATA_PATCH_TEST/All





To create the database, you must first create the following directories: `FEATURES/DATABASE`. Then, you should have the `sampled_metadata.csv` file from the `PathologySearchComparison/data/DATABASE`. In the following example, we assume you have copied this `.csv` file to the `FEATURES/DATABASE` directory.
```bash
python parallel_patching.py --data_dir PathologySearchComparison/DATA/DATABASE/ --metadata_path ./FEATURES/DATABASE/sampled_metadata.csv --experiment DATABASE --save_dir ./FEATURES/DATABASE/PATCHES --num_processes 16
```
This code will create a folder called `PATCHES` in `FEATURES/DATABASE`. Then, you should run the following command to create the features from the extracted patches:
```bash
python feature_extraction.py --patch_dir ./FEATURES/DATABASE/PATCHES/ --experiment DATABASE --extracted_features_save_adr ./FEATURES/DATABASE/features.pkl --batch_size 256 --use_gpu True
```
This code will save the extracted features as `features.pkl` in `FEATURES/DATABASE`. For the case of our experiment, the code was not able to extract features of 6 slides which are presented in `FEATURES/DATABASE/not_processed.csv`. We used slurm resource management for running these codes. The commands as well as logs of running these commands can be found in `FEATURES/DATABASE/scripts/`.

To create the Database for the ablation study, you just need to use `ablation_feature_extraction.py` instead of `feature_extraction.py`. <ins>Please be advised that here, the term *ablation* is referred to the original DenseNet feature extractor. Commands that do not have *ablation* in their name are using the KimiaNet feature extractor. Please excuse us for being counterintuitive.</ins>
```bash
python ablation_feature_extraction.py --patch_dir ./FEATURES/DATABASE/PATCHES/ --experiment ABLATION_DATABASE --extracted_features_save_adr ./FEATURES/ABLATION_DATABASE/features.pkl --batch_size 256 --use_gpu True
```

## Test Data Featrue Extraction
Here, we would essentially follow the same procedure as above to patch and extract the features from test slides. Except for the UCLA and reader study, other experiments do not study patch level retrieval. For examle, for the Yale HER2 experiment, these are the commands that need be run:
```bash
python parallel_patching.py --data_dir PathologySearchComparison/DATA/TEST_DATA/BRCA_HER2/ --metadata_path ./FEATURES/TEST_DATA/BRCA_HER2/test_metadata.csv --experiment BRCA_HER2 --save_dir ./FEATURES/TEST_DATA/BRCA_HER2/PATCHES --num_processes 8
python feature_extraction.py --patch_dir ./FEATURES/TEST_DATA/BRCA_HER2/PATCHES/ --experiment BRCA_HER2 --extracted_features_save_adr ./FEATURES/TEST_DATA/BRCA_HER2/features.pkl --batch_size 256 --use_gpu True
```
For each test set, all of the commands required ti generate features can be found in the corresponding subdirectories for each test set (e.g. `FEATURES/TEST_DATA/BRCA_HER2/scripts`).

For UCLA and reader study, you need to run an additional step to get the results for patch retrieval. For example:
```bash
python parallel_patching.py --data_dir PathologySearchComparison/DATA/TEST_DATA/READER_STUDY/ --metadata_path ./FEATURES/TEST_DATA/READER_STUDY/test_metadata.csv --experiment READER_STUDY --save_dir ./FEATURES/TEST_DATA/READER_STUDY/PATCHES --num_processes 8
python feature_extraction.py --patch_dir ./FEATURES/TEST_DATA/READER_STUDY/PATCHES/ --experiment READER_STUDY --extracted_features_save_adr ./FEATURES/TEST_DATA/READER_STUDY/features.pkl --batch_size 256 --use_gpu True
python feature_extraction.py --patch_dir ./FEATURES/TEST_DATA/READER_STUDY/QUERY_PATCHES/ --experiment READER_STUDY --extracted_features_save_adr ./FEATURES/TEST_DATA/READER_STUDY/query_features.pkl --batch_size 256 --use_gpu True
```

## Searching the Database for Test Slides
Once you extract the features as outlined above, you can use the `search_template.ipynb` notebook located at `TEST_DATA_RESULTS` to carry out the search and store the performance metrics. For our experiments, we have provided the notebook we have used to store the data in each corresponding subdirectory (e.g. `TEST_DATA_RESULTS/ABLATION_BRCA_HER2/search.ipynb`).

## Runtime Analysis
We have also provided the code we have used to evaluate the runtime of the model for the Yale Trastuzumab study under `TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/` directory. In the `scripts` directory, you can find the logs for patching and feature extraction, and in the accompanied notebook, you can find the search and retrieval runtimes.