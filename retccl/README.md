# Running RetCCL
Before running the codes, make sure that you first change the directory to this directory:
```bash
cd retccl
```
To run the retccl framework, we must first create the database index, then calculate test data features, and finally, carrying out the search.

## Creating Database
The first step to create the database index, is to patch the slides. In this method, we should run seperate commands for each diagnosis. For example, for LGG and GBM, the following commands are used:
```bash
python create_patches_fp.py --source PathologySearchComparison/DATA/DATABASE/brain/LGG --seg --patch --stitch --save_dir ./FEATURES/DATABASE/brain/LGG --preset DATABASE.csv
python create_patches_fp.py --source PathologySearchComparison/DATA/DATABASE/brain/GBM --seg --patch --stitch --save_dir ./FEATURES/DATABASE/brain/GBM --preset DATABASE.csv
```
The next step would be generating the `patch_dataframe.csv`. To do so, just go to `FEATURES/DATABASE/patch_dataframe.ipynb` and run all cells. Make sure that `sampled_metadata.csv` file from the `PathologySearchComparison/data/DATABASE` is copied to the same directory. Once this `.csv` is generated, you can extract the features by running the following command:
```bash
python extract_features.py --patch_dataframe_path ./FEATURES/DATABASE/patch_dataframe.csv --save_dir ./FEATURES/DATABASE
```
This command will generate `features.h5`. Finally, you can generate the mosaics by running this command:
```bash
generate_mosaics.py --features_path ./FEATURES/DATABASE/features.h5 --save_dir ./FEATURES/DATABASE --kl 9 --R 0.2
```
This will generate `features_with_cluster.h5` and `mosaics.h5` used for searching.


## Test Data Featrue Extraction
Just like the database slides, for each test set we should start by patching the slides. For example, for the `BRCA_HER2`, we can run:
```bash
python create_patches_fp.py --source PathologySearchComparison/DATA/TEST_DATA/BRCA_HER2/ --seg --patch --stitch --save_dir ./FEATURES/TEST_DATA/BRCA_HER2 --preset BRCA_HER2.csv
```
The next step is optional, and is only used for the patch retrieval purposes. You can visualizing every patches resulted from `create_patches_fp` and saving them to file by running:
```bash
python all_patches_visualization.py --slides_dir PathologySearchComparison/DATA/TEST_DATA/BRCA_HER2 --patches_dir ./FEATURES/TEST_DATA/BRCA_HER2/patches --save_dir ./FEATURES/TEST_DATA/BRCA_HER2/visualized_patches --slide_extension svs
```
Like the procedure for database, we must generate the `patch_dataframe.csv`. To do so, for each test, just run all the cells in `FEATURES/TEST_DATA/[test set]/patch_dataframe.ipynb`. Once this `.csv` is generated, you can extract the features by running the following command:
```bash
python extract_features.py --patch_dataframe_path ./FEATURES/TEST_DATA/BRCA_HER2/patch_dataframe.csv --save_dir ./FEATURES/TEST_DATA/BRCA_HER2
```
This will generate `features.h5`. Then, you can generate the mosaics by running this command:
```bash
python generate_mosaics.py --features_path ./FEATURES/TEST_DATA/BRCA_HER2/features.h5 --save_dir ./FEATURES/TEST_DATA/BRCA_HER2 --kl 9 --R 0.2
```
This command will generate `features_with_cluster.h5` and `mosaics.h5`.

## Searching the Database for Test Slides
To perform the search and find the metrics, for each experiment, simply run `search.py`. For example:
```bash
python search.py --experiment BRCA_HER2
```
This will create the following files for each test slide: `Bags.pkl`, `Entropies.pkl`, `Etas.pkl`, and `Results.pkl`. These files will be located at `FEATURES/TEST_DATA/[test set]/results`. Then, you can use the `search_template.ipynb` notebook located at `TEST_DATA_RESULTS` to carry out the search and store the performance metrics.

## Runtime Analysis
We have also provided the code we have used to evaluate the runtime of the model for the Yale Trastuzumab study under `TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/` directory. In the `scripts` directory, you can find the logs for patching and feature extraction. One runtime is considered the sum of runtimes of `run{i}.sbatch` and `search{i}.sbatch`.