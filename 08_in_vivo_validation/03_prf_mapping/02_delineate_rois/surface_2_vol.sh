SUB="$1"
HEMI="$2h"
SUBJECT_ID="sub-0$SUB"
SUBJECT_ID_2="sub0$SUB"

# Convert labels from surface space to volume space

echo "Converting manually-defined ROI labels to niftis in functional space for subject $SUBJECT_ID"

# Define directories
PROJECT_DIR=../relational_neural_control/in_vivo_validation/in_vivo_fmri_dataset

# Set where to look for the segmentation/surface files (needed for the "proj" step)
export SUBJECTS_DIR=$PROJECT_DIR/data_FS

## Define the ROI labels 
declare -a ROIs=("V1" "V4")

for ROI in ${ROIs[@]}; do

    # convert manually-defined label to nifti (func space)
    mri_label2vol --label $PROJECT_DIR/data_FS/$SUBJECT_ID/label/$HEMI.${ROI}.label \
    --temp $PROJECT_DIR/derived/$SUBJECT_ID_2/alldata/run01/uarf0${SUB}-01_mean.nii \
    --regheader $PROJECT_DIR/data_FS/$SUBJECT_ID/mri/orig.mgz \
    --subject $SUBJECT_ID \
    --hemi $HEMI \
    --o $PROJECT_DIR/data_FS/$SUBJECT_ID/label/$HEMI.$ROI.func.nii.gz \
    --proj frac 0 1 .1

done
