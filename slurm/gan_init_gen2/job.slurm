#!/bin/bash -xl
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --output=./slurm/logs/%x.%j.log


unset SLURM_EXPORT_ENV

export https_proxy=http://proxy:80
export http_proxy=http://proxy:80

# Get the directory where this script is located
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

module load python/3.12-conda
module load cuda/12.6

conda activate bube 
python3 "$SLURM_SUBMIT_DIR/slurm/gan_init_gen2/script.py" "${@:1}"