#!/bin/bash

#SBATCH --job-name=fs_reconall
#SBATCH --mail-user=email
#SBATCH --mail-type=end
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=5
#SBATCH --time=10:00:00
#SBATCH --qos=prio

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo omp num threads: $OMP_NUM_THREADS

# set subject id
SUBJECTS=('sub-01' 'sub-02' 'sub-03' 'sub-04' 'sub-05' 'sub-06')
SUBJECT_ID=${SUBJECTS[${SLURM_ARRAY_TASK_ID}]}

# set which FS command to run
which_FS=reconall

echo "Running $SUBJECT_ID recon-all."

# load modules
module purge
module load FreeSurfer/7.3.2-centos7_x86_64 ; source $FREESURFER_HOME/SetUpFreeSurfer.sh

# set subject directory directory for outputing new surfaces
SUBJECTS_DIR=../relational_neural_control/in_vivo_validation/in_vivo_fmri_dataset/data_FS/
T1_IMG=../relational_neural_control/in_vivo_validation/in_vivo_fmri_dataset/raw/${SUBJECT_ID}/ses-01/anat/${SUBJECT_ID}_ses-01_T1w.nii # Download raw fMRI from: https://openneuro.org/datasets/ds005503

# run chosen FS command
if [ "$which_FS" = "reconall" ]; then
    recon-all -all -subjid $SUBJECT_ID -sd $SUBJECTS_DIR -i ${T1_IMG} -openmp $OMP_NUM_THREADS
elif [ "$which_FS" = "pial" ]; then
    recon-all -autorecon-pial -no-isrunning -openmp $OMP_NUM_THREADS -nomaskbfs -subjid $SUBJECT_ID -sd $SUBJECTS_DIR
fi
