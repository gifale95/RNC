SUBJECT_ID="subj0${1}"
HEMI="$2h"

FS_DIR=../natural-scenes-dataset/nsddata/freesurfer/$SUBJECT_ID

freeview -f $FS_DIR/surf/$HEMI.inflated:overlay=$FS_DIR/label/$HEMI.prfeccentricity.mgz:overlay_color='colorwheel':overlay=$FS_DIR/label/$HEMI.prfangle.mgz:overlay_custom=./${HEMI}_pol:overlay=$FS_DIR/label/$HEMI.prfsize.mgz:overlay_color='colorwheel':overlay=$FS_DIR/label/$HEMI.prf-visualrois.mgz:overlay_color='colorwheel'
