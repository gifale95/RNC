%% script for writing MNI EVC mask from group level contrast

%setup some variables 

derived_dir = '/scratch/singej96/dfg_projekt/WP1/derived/exp/roi';

disp('Write LOC mask in MNI space')
% write mask for LOC
locL_hdr = spm_vol(fullfile(derived_dir,'object_parcels',['lLOC.img']));
locL_vol = spm_read_vols(locL_hdr);
locR_hdr = spm_vol(fullfile(derived_dir,'object_parcels',['rLOC.img']));
locR_vol = spm_read_vols(locR_hdr);

loc_vol = locL_vol;
loc_vol(locR_vol~=0) = 1; 

loc_hdr = locL_hdr;
loc_hdr.fname = fullfile(derived_dir,'loc_mni.nii');
loc_hdr.descrip = 'Kanwisher loc defintion';
loc_hdr.dt = [2 0];
loc_hdr.private.dat.dtype = 'FLOAT32-LE';
loc_hdr.private.dat.scl_slope = 1;
loc_hdr.private.dat.scl_inter = 0;
spm_write_vol(loc_hdr,loc_vol);
