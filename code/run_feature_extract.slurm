#!/bin/bash
#SBATCH --job-name=featureextract
#SBATCH --output=logs/features_full_%j.out
#SBATCH --error=logs/features_full_%j.err
#SBATCH --partition=compute
#SBATCH --account=sscm037184
#SBATCH --time=04:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Create logs directory
mkdir -p logs

# Load Conda and activate environment
source ~/initConda.sh
conda activate summative_env

# Run the conversion
echo "Starting conversion at $(date)"
#python convert_to_nifty.py
echo "Finished conversion at $(date)"

# Run feature extraction
echo "Starting feature extraction at $(date)"
python feature_extract.py
echo "Finished feature extraction at $(date)"


echo "Job finished at: $(date)"
