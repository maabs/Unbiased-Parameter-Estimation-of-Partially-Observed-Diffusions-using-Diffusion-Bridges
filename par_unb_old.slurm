#!/bin/bash
#SBATCH --array=1-5
#SBATCH --partition=batch
#SBATCH -J test
#SBATCH -o displays/test.%A.%a.out
#SBATCH -e displays/test.%A.%a.err
#SBATCH --mail-user=miguelangel.alvarezballesteros@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=72:00:00
#SBATCH --mem=50G
#SBATCH --ntasks=5
#SBATCH --ntasks-per-node=1


#run the application:
#source ~/miniconda3/bin/activate loc 
export CONDA_PKGS_DIRS=/ibex/users/alvarem/conda_cache
source /ibex/user/alvarem/miniconda3/bin/activate conda activate /home/alvarem/miniconda3/envs/loc
Array=("2" "2" "2" "2" "2") 
ARGUMENT=${Array[$SLURM_ARRAY_TASK_ID-1]}
python Par_loc.py "$ARGUMENT"
