# Downloading and Preparing the Data
The process of preparing the datasets we used in this study has two main steps:
1. Downloading and preparing the search databes slides.
2. Downloading and preparing the individual test datasets.

## Search Database
To download slides from the [GDC Data Portal](https://portal.gdc.cancer.gov/), you need to make sure you have [GDC Data Transfer Tool](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) installed. This tool requires a manifest file to download the set of slides we are interested in.

In the `DATABASE` subdirectory, you will find four jupyter notebooks:
1. `gdc_search.ipynb`: This notebook will query ALL `.svs` files in the GDC Data Portal with `primary_site` being in th `["Breast", "Brain", "Bronchus and lung", "Colon", "Liver and intrahepatic bile ducts"]`. It will then rename fields of interest to a more human friendly format, and saves the results in `metadata.csv`.
2. `sample.ipynb`: This notebook implements our sampling logic for randomly selecting a smaller set to play the role of our database slides. A random seed is provided for reproducability. The result is saved in `sampled_metadata.csv`.
3. `manifest.ipynb`: This notebook prepares the `manifest.txt` file required to download the slides.
4. `organize.ipynb`: Once the manifest file is created, you can use the code in `download.sbatch` to download the slides. The location to store the downloaded files, as well as other downlaod parameters are stored in `dtt-config.dtt`. Assuming the slides are downloaded to `PathologySearchComparison/DATA/DATABASE`, you can use `organize.ipynb` to organize the downloaded slides into the following structure:
```bash
DATA
└── DATABASE
    ├── PRIMARY_SITE
    │   ├── DIAGNOSIS
    │   │   ├── slide_1
    │   │   ├── slide_2
    │   │   ├── slide_3
```

## Test Datasets
In this study, we have used 6 different datasets for different studies. Except for the in-house data (UCLA) that can be downloaded from Zenodo repository, all other datasets can be downloaded from Cancer Imaging Archive.
1. [Three in-house slides (UCLA)](https://zenodo.org/records/10835156)
2. [CMB-CRC](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93257955)
3. [CMB-LCA](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=93258420)
4. [Yale HER2 and Trastuzumab](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=119702524)
5. [CPTAC-GBM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=30671232)
6. [UPENN-GBM](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70225642)

An overview of the statistics of these datasets can be found in `TEST_DATA/dataset_statistics.ipynb`. For the reader study, we have randomly selected one slide from each of the `CMB-CRC`, `CMB-LCA`, and `Yale HER2`. The code for this selection can be found in `TEST_DATA/READER_STUDY/reader_study.ipynb`. Also, for the microscope study, we have randomly selected 34 slides out of 178 `CPTAC-GBM` slides. The code for this selection can be found at `TEST_DATA/GBM_MICROSCOPE/gbm.ipynb`. Since `Yale HER2` and `Yale Trastuzumab` datasets are already seperated when downloaded from the source, not furtehr preparation is needed.

Once these datasets are downloaded and the required slides are chosen, you should arrange them in a similar structure as above.

## Final Structure of the DATA Directory
Once all datasets are prepared, the `PathologySearchComparison/DATA` directory should look like this:
```bash
DATA
├── DATABASE
│    ├── PRIMARY_SITE
│    │   ├── DIAGNOSIS
│    │   │   ├── slide_1
│    │   │   ├── slide_2
│    │   │   ├── slide_3
├── TEST_DATA
│    ├── UCLA
│    │    ├── slide_1
│    │    ├── slide_2
│    │    ├── slide_3
│    ├── READER_STUDY
│    │    ├── slide_1
│    │    ├── slide_2
│    │    ├── slide_3
│    ├── BRCA_HER2
│    │    ├── slide_1
│    │    ├── slide_2
│    │    ├── slide_3
│    ├── BRCA_TRASTUZUMAB
│    │    ├── slide_1
│    │    ├── slide_2
│    │    ├── slide_3
│    ├── GBM_MICROSCOPE_CPTAC
│    │    ├── slide_1
│    │    ├── slide_2
│    │    ├── slide_3
│    ├── GBM_MICROSCOPE_UPENN
│    │    ├── slide_1
│    │    ├── slide_2
│    │    ├── slide_3
```

## Notes
For the ablation studies, for latent variables not to be mixed up with each other, we might temporarily rename these datasets by adding the `ABLATION_` prefix to the the directory names. Also, for methods like SISH, we are required to have the directories being structured like this:
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
We will provide code to change the structure and revert it back whenever necessary.