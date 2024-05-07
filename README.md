# Pathology Search Comparison
This pipeline is comprised of three main steps:
1. Downloading and storing database and test datasets.
2. Running each of the four models' to create model specific database and generating results.
3. Post-Hoc analysis of the results.

To reproduce our results, you can follow the guidelines presented here. Also, please feel free to use our pipeline to test these models on you custom database and test datasets.

Throughout this repository, you will often find folders with `.gitignore` inside them. These folders are just placeholders to demonstrate how the directory often looks like after you run the model. Due to the large size of these compiled data, it was not possible to include them in this repository. However, they are available upon reasonable request.

## Setting Up the Conda Environment:
We recommend having [Mamba](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) installed. You can run the following command to install all the dependencies:
```bash
mamba/conda env create -f search.yml
```

## Downloading Data
We have provided dtailed guidelines on how to download the TCGA slides we used as a common database, and test slides we used for different studies in the `data_preparation` subdirectory of this repository. Please refer to `data_preparation/README.md` for detailed explanation.

## Running the Models
For each of the four methods we compared in this study, we have provided the source code we used to generate the results in this repository in the following subdirectories:
1. `yottixel`
2. `sish`
3. `retccl`
4. `hshr`

Inside each one of these subdirectories, you can find a detailed explanation on how to run these models in their corresponding `README.md` file.

## Post-Hoc Analysis
The final aggregation and analysis of the results is performed using scripts provided here in the `analysis` subdirectory. A brief explanation of how to run these scritps is provided in `analysis/README.md`.

## Refrences
This study has evaluated the following refrences for readiness to be deployed in the clinical settings:
1. [Yottixel](https://github.com/KimiaLabMayo/yottixel/tree/main)
2. [SISH](https://github.com/mahmoodlab/SISH/tree/main)
3. [RetCCL](https://github.com/Xiyue-Wang/RetCCL)
4. [HSHR](https://github.com/Lucius-lsr/HSHR)
