#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH -J test
#SBATCH -o displays/test.%J.out
#SBATCH -e displays/test.%J.err
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=23:59:00
#SBATCH --mem=100G
#SBATCH --ntasks=40
#SBATCH --ntasks-per-core=1

#run the application:
#source ~/miniconda3/bin/activate loc 
export CONDA_PKGS_DIRS=/ibex/users/alvarem/conda_cache
source /ibex/user/alvarem/miniconda3/bin/activate conda activate /home/alvarem/miniconda3/envs/loc

python Par_loc.py
