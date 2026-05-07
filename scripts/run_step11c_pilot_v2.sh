#!/bin/bash
#SBATCH --job-name=ddi_step11c_v2
#SBATCH --account=def-cottenie
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/step11c_v2_%j.out
#SBATCH --error=logs/step11c_v2_%j.err

source $HOME/ddiproject_setup.sh
cd /scratch/vlelo/ddiproject

python scripts/step11c_pathway_retrieval_v2.py \
    --data-dir /scratch/vlelo/ddiproject/processed_v2 \
    --out-dir /scratch/vlelo/ddiproject/processed_v2 \
    --target test \
    --pilot 500 \
    --asymmetric-penalty 0.5
