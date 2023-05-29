#!/bin/bash
#

#SBATCH --job-name=CreateNetwork
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6g
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michell.queiroz@uantwerpen.be
#SBATCH -o stdout.file
#SBATCH -e stderr.file

module load calcua/supported
conda activate ox

python REQreate.py
