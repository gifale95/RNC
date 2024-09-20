SUBJECT_ID="sub-0${1}"
HEMI="$2h"

PRF_MODEL=prf_Iso_fit_hrf_False

FS_DIR=../relational_neural_control/in_vivo_validation/in_vivo_fmri_dataset/data_FS/$SUBJECT_ID
PRFPY_DIR=../relational_neural_control/in_vivo_validation/in_vivo_fmri_dataset/prf_output/${PRF_MODEL}

freeview -f $FS_DIR/surf/$HEMI.inflated:overlay=$PRFPY_DIR/$SUBJECT_ID/$HEMI.ecc.mgh:overlay_color='colorwheel':overlay=$PRFPY_DIR/$SUBJECT_ID/$HEMI.pol.mgh:overlay_custom=./${HEMI}_pol:overlay=$PRFPY_DIR/$SUBJECT_ID/$HEMI.prf_size.mgh:overlay_color='colorwheel'
