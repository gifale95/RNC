#!/bin/bash

#SBATCH --job-name=prf_mapping_sub-1
#SBATCH --mail-user=email
#SBATCH --mail-type=end
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=25
#SBATCH --time=4-00:00:00
#SBATCH --qos=standard

declare -a combinations
index=0
for sub in 0
do
    for hem in 0 1
    do
        combinations[$index]="$sub $hem"
        index=$((index + 1))
    done
done

parameters=(${combinations[${SLURM_ARRAY_TASK_ID}]})

SUB_ID=${parameters[0]}
HEM_ID=${parameters[1]}
# 0 - lh, 1 - rh

#Print out parameters
echo sub $SUB_ID
echo hem $HEM_ID

# activate conda environment
# include for this reason: https://stackoverflow.com/questions/34534513/calling-conda-source-activate-from-bash-script
# You can install this environment using the .yml file in:
# "https://github.com/gifale95/RNC/06_in_vivo_validation/03_prf_mapping/01_prf_mapping/prf-workflow.yml"
eval "$(conda shell.bash hook)"
conda activate prf-workflow

# load modules
module purge
module load MATLAB/2021a
module load FreeSurfer/7.3.2-centos7_x86_64 ; source $FREESURFER_HOME/SetUpFreeSurfer.sh
module load FSL/5.0.11-foss-2018b-Python-3.6.6
module add ANTs/2.3.1-foss-2018b-Python-3.6.6
module add AFNI/18.3.00-foss-2018b-Python-3.6.6

# get environment variable of num threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SLURM_JOB_ID=$SLURM_JOB_ID

# unset Java max heap space option
unset _JAVA_OPTIONS

# Change directory to the prf-workflow package
# https://github.com/mayajas/prf-workflow
cd ../prf-workflow/prf-workflow/

# run the thing
CONFIG_FILE=../06_in_vivo_validation/03_prf_mapping/01_prf_mapping/prf_mapping_config_s1.json
echo $CONFIG_FILE
# Add your anaconda root directory
../anaconda3/envs/prf-workflow/bin/python pRF_mapping.py $CONFIG_FILE $SUB_ID $HEM_ID

