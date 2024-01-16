#!/bin/bash
#

#SBATCH --job-name=CreateNetwork
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --mem-per-cpu=6g
#SBATCH --time=72:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michell.queiroz@uantwerpen.be
#SBATCH -o stdout3.file
#SBATCH -e stderr3.file
#SBATCH --partition=skylake

module load calcua/supported
source $VSC_DATA/miniconda3/etc/profile.d/conda.sh
conda activate ox

python REQreate3.py
