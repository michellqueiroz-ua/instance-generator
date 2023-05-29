#!/bin/bash
#

#SBATCH --jobname=create_network
#SBATCH --ntasks=1 –-cpus-per-task=8
#SBATCH –-mem-per-cpu=6g
#SBATCH --time=06:00:00
#SBATCH --mail-type=ALL
#SBATCH -–mail-user=michell.queiroz@uantwerpen.be
#SBATCH -o stdout.file
#SBATCH -e stderr.file

conda activate ox

python REQreate.py
