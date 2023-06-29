# Pathology Search Comparison

## database
Here, you can find two notebooks: "gdc_search.ipynb" and "sample.ipynb". "gdc_search.ipynb" uses NCI's GDC API to retrieve all the .SVS slides in TCGA project that have a `primary_site` of rither "Breast", "Brain", "Bronchus and lung", "Colon", or "Liver and intrahepatic bile ducts". "sample.ipynb" would randomly sample 50 to 75 slides from each category to create the dataset that would be used to create the databases of each method. This sampled dataset is stored as "sampled_metadata.csv".

## Yottixel

```python
python feature_extraction.py --batch_size 256
```

```python
python parallel_patching.py --metadata_path ./sampled_metadata_okay.csv --num_processes 16
```
## RetCCL

```python
python feature_extraction.py --batch_size 256
```

```python
python parallel_patching.py --metadata_path ./sampled_metadata_okay.csv --num_processes 16
```