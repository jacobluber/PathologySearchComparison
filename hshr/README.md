# Running HSHR
Before running the codes, make sure that you first change the directory to this directory:
```bash
cd hshr
```
To run the hshr framework, we must first create the database index, then calculate test data features, and finally, carrying out the search.

## Creating Database
Just like the other methods, we should start by preprocessing the slids. This can be done by running `preprocess.py` for every primary site individually. The following example is for the brain site:
```bash
python preprocess.py --WSI_DIR PathologySearchComparison/DATA/DATABASE/DATABASE/brain --SLIDE_EXT svs --SSL_RESNET_MODEL_DICT ./checkpoints/model_best.pth --RESULT_DIR ./FEATURES/DATABASE/brain --TMP ./FEATURES/DATABASE/TEMP
```
Once this code is run for all sites, we must update and move `FEATURES/DATABASE/CONST.py` to root (`hshr/`) before running the SSL encoder using the command below:
```bash
python ssl_encoder_training.py --RESULT_DIR ./FEATURES/DATABASE --TMP ./FEATURES/DATABASE/TEMP --MODEL_DIR ./FEATURES/DATABASE/model --DATASETS brain breast colon liver lung
```
This code will generate `model_best.pth` and save it under `FEATURES/DATABASE/model/ssl_att`. Additionally, we have provided the list of slides that we were not able to process using this pipeline in `FEATURES/DATABASE/not_processed.txt`.

Optionally, you can train your backbone using your data by running the following commands. However, for this study, we used the authors' original backbone.
```bash
cd hshr/backbone
python generate.py --SVS_DIR PathologySearchComparison/DATA/DATABASE/ --RESULT_DIR ../FEATURES/DATABASE/BACKBONE/ --TMP ../FEATURES/DATABASE/TEMP_BACKBONE/ --SIZE 1000
```
```bash
cd backbone/SimCLR
python run.py --pathology_root ../../FEATURES/DATABASE/BACKBONE --save_path ../../FEATURES/DATABASE/BACKBONE_RESULTS/ 
```

## Test Data Featrue Extraction
To extract the features for each test set, we can similarily run the following command for each test set:
```bash
python preprocess.py --WSI_DIR PathologySearchComparison/DATA/TEST_DATA/BRCA_HER2 --SLIDE_EXT svs --RESULT_DIR FEATURES/TEST_DATA --TMP FEATURES/TEST_DATA/TEMP 
```
Just be cautious about the extensions of the slides in different sets:
```bash
python preprocess.py --WSI_DIR PathologySearchComparison/DATA/TEST_DATA/GBM_MICROSCOPE_UPENN --SLIDE_EXT ndpi --RESULT_DIR FEATURES/TEST_DATA --TMP FEATURES/TEST_DATA/TEMP 
```

## Searching the Database for Test Slides
You can run `search.ipynb` notebook located in the root of `hshr` directory to perform the hypergraph-based retrieval and calculate the performance metrics.

## Runtime Analysis
We have also provided the code we have used to evaluate the runtime of the model for the Yale Trastuzumab study under `TEST_DATA_RESULTS/TIME_BRCA_TRASTUZUMAB/` directory. In the `scripts` directory, you can find the logs for patching and feature extraction, and in the accompanied notebook, you can find the search and retrieval runtimes.
