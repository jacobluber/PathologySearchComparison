# Pathology Search Comparison
In order to run each of the models, you should change the working directory to the corresponding directory (e.g. `cd yottixel`).

## database
Here, you can find two notebooks: "gdc_search.ipynb" and "sample.ipynb". "gdc_search.ipynb" uses NCI's GDC API to retrieve all the .SVS slides in TCGA project that have a `primary_site` of rither "Breast", "Brain", "Bronchus and lung", "Colon", or "Liver and intrahepatic bile ducts". "sample.ipynb" would randomly sample 50 to 75 slides from each category to create the dataset that would be used to create the databases of each method. This sampled dataset is stored as "sampled_metadata.csv".

## Yottixel
First you have to generate the mosaics from database slides and test slides. To do so, you can use either "patching.py" or "parallel_patching.py". The only difference is that the latter uses parallel CPUs for a faster runtime.

```python
python patching.py --data_dir [TCGA_DATA_DIR] --metadata_path ./sampled_metadata.csv --save_dir ./PATCHES
python parallel_patching.py --data_dir [TCGA_DATA_DIR] --metadata_path ./sampled_metadata.csv --save_dir ./PATCHES --num_processes 16
```

```python
python feature_extraction.py --batch_size 256
```


## RetCCL

```python
python feature_extraction.py --batch_size 256
```

```python
python parallel_patching.py --metadata_path ./sampled_metadata_okay.csv --num_processes 16
```