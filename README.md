# University of Bristol
## MSc Medical Statistics and Health Data Science
### Artificial Intelligence for Health Data Science
#### 2623764, May 2026

## Research Aim

This project explores predictive modelling comparisons between XGBoost, a traditional machine learning random forest approach, and multi-layer perceptron, a neural network approach. Different feature sets combining clinical, genomic and radiomic data are evaluated to predict survival outcomes at 3-years post lung cancer diagnosis.

---

## Notes

- This project is unfortunately not able to be run via a pipeline in HPC.
- This project was completed without Git software, and thus the git repository includes only commits via the GitHub website.

## Project Directory Structure

- **Summative**: Root directory of the project.
- **data/**: Contains all data.
  - **Raw/**: Raw data downloaded from PubMed.
  - **Derived/**: Processed data.
- **Output/**: Outputs created from analysis of processed data.
- **Code/**: All scripts used to conduct the work.
- **logs/**: Directory to store log notes from ran scripts.

---


## How to set up (if code in a suitable format to run via HPC in a pipeline)

**Log into BluePebble (HPC)**

**Set up the Conda environment**

```
conda env create --file environment.yml
```

or if the environment already exists

Conda must be initialised on the server using

```
source ~/initConda.sh
```

```
conda env update --file environment.yml --prune
```

## How to run the pipeline

**Activate Conda environment**

```
conda activate summative_env
```

**Run the analysis pipeline using Snakemake**

```
snakemake --profile .
```

And this requires a `config.yaml file`, with the SnakeMake version 7.26.

---

## Code and Data Summary

- **`convert_to_nifty.py`**
  - Scans patient session directories to locate CT DICOM series and DICOM SEG files by reading modality tags from file headers.
  - Converts CT DICOM series to compressed NIfTI (.nii.gz) using dicom2nifti.
  - Converts DICOM SEG files to NIfTI using the dcmqi command-line tool, then compresses the output to .nii.gz.
  - Processes all patients and outputs to separate image and mask directories.

- **`feature_extract.py`**
  - Loads and combines multiple segmentation files per patient into a single binary mask, resampling to match CT dimensions using nearest-neighbour interpolation.
  - Extracts 6 shape features from tumour surface meshes generated via marching cubes
  - Extracts 6 intensity features from Hounsfield Unit values within the tumour region and 3 GLCM texture features using 13 directional offsets and 32 grey levels.
  - Outputs three CSV files indexed by patient ID

- **`data_processing - clinical and RNA.R`**
  - Reads in raw clinical and gene expression data
  - Preprocesses the data to clean and factorise categorical variables
  - Remove data which is considered data leakage
  - Combined the genomic data with the clinical data.
  - Data is split and imputed using K nearest-neighbour algorithm

- **`data_processing - images.R`**
  - Reads in raw image data created from `feature_extract.py`
  - Processing the data combining it into one dataset and merging it with the raw clinical data
  - Data is split and imputed using K nearest-neighbour algorithm

- **`XGBoost_analysis.R`**
  - Data is processed and hyperparameters are tuned
  - Models are fit based on the optimal hyperparameters
  - Outcome metrics are presented via tables and figures
  - Oversampling techniques are implemented

- **`Neural_network_analysis.R`**
  - Data is processed and hyperparameters are tuned
  - Models are fit based on the optimal hyperparameters
  - Outcome metrics are presented via tables and figures
  - PFI analysis is implemented

---
