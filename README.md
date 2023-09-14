# Pathology Search Comparison
In order to run each of the models, you should change the working directory to the corresponding directory (e.g. `cd yottixel`).

## database
Here, you can find two notebooks: `gdc_search.ipynb` and `sample.ipynb`. `gdc_search.ipynb` uses NCI's GDC API to retrieve all the .SVS slides in TCGA project that have a `primary_site` of rither "Breast", "Brain", "Bronchus and lung", "Colon", or "Liver and intrahepatic bile ducts". `sample.ipynb` would randomly sample 50 to 75 slides from each category to create the dataset that would be used to create the databases of each method. This sampled dataset is stored as `sampled_metadata.csv`.

## Yottixel
First you have to generate the mosaics from database slides and test slides. To do so, you can use either `patching.py` or `parallel_patching.py`. The only difference is that the latter uses multiprocessing for a faster runtime.

```python
python patching.py --data_dir [TCGA_DATA_DIR] --metadata_path './sampled_metadata.csv' --save_dir './PATCHES'
```
Or
```python
python parallel_patching.py --data_dir [TCGA_DATA_DIR] --metadata_path './sampled_metadata.csv' --save_dir './PATCHES' --num_processes 16
```

Next step would be to use KimiaNet to extract features from each patch in the mosaics. This can be achieved by running: 
```python
python feature_extraction.py --patch_dir './PATCHES/' --extracted_features_save_adr './extracted_features.pickle' --batch_size 256 --use_gpu True
```
This will save the extracted features to `extracted_features.pickle`.

For the slides in the test set, you should follow the same two steps to calculate and save `extracted_test_features.pickle`. The rest of the process can be followed in `search.ipynb` notebook.

## RetCCL

The first step here is to create patches. The authors have used to same patching pipeline as SISH. Hence, the data should be stored in the following format:
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
Then, for `20x` resolution slides you can run the following command. Make sure that all your presets for segmentation and patching are stored in a `.csv` file inside the `presets` directory.
```python
python create_patches_fp.py --source ./DATA/WSI/[[SITE]/[DIAGNOSIS]/20x/ --step_size 1024 --patch_size 1024 --patch_level 0 --seg --patch --stitch --save_dir ./DATA/PATCHES/[SITE]/[DIAGNOSIS]/20x --preset tcga.csv
```
And for `40x` resolution slides you can run: 
```python
python create_patches_fp.py --source ./DATA/WSI/[[SITE]/[DIAGNOSIS]/40x/ --step_size 512 --patch_size 512 --patch_level 1 --seg --patch --stitch --save_dir ./DATA/PATCHES/[SITE]/[DIAGNOSIS]/40x --preset tcga.csv
```
Once the patches are created, you have to extract features for each patch before generating the mosaics. To do so, you have to create a dataframe called `patch_dataframe.csv` file with the following columns: `file_id`, `file_name`, `slide_path`, `patch_level`, `patch_size`, `coord1`, and `coord2`.
```python
python extract_features.py
```
