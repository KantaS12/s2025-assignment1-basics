#!/bin/bash
#SBATCH --partition=shared
#SBATCH --job-name=bpe_train
#SBATCH --output=bpe_results.log
#SBATCH --error=bpe_error.log

#SBATCH --mem=64G                # Request 64GB of RAM
#SBATCH --time=18:00:00          # Set limit to 18 hours
#SBATCH --cpus-per-task=16       # Use 16 cores (helps the initial word counting)

cd /home/kantas/ece405-assignment1-basics
source ECE405/bin/activate

python3 cs336_basics/bpe_tokenizer.py