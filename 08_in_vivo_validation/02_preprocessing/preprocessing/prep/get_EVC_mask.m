%% script for writing MNI EVC mask from group level contrast

%setup some variables 

derived_dir = '/scratch/singej96/dfg_projekt/WP1/derived/exp/roi';

disp('Write EVC mask in MNI space')
% write mask for V1-V3 (still in MNI space)
EVC_ROI = [1, 4, 5];
hcp_hdr = spm_vol(fullfile(derived_dir,['HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii']));
hcp_vol = spm_read_vols(hcp_hdr);

evc_hdr = hcp_hdr;
evc_hdr.fname = fullfile(derived_dir,'evc_mni.nii');
evc_hdr.descrip = 'Glasser atlas with regions V1, V2, V3';
evc_hdr.dt = [2 0];
evc_hdr.pinfo = [1 0 352]';
evc_hdr.private.dat.dtype = 'FLOAT32-LE';
evc_hdr.private.dat.scl_slope = 1;
evc_hdr.private.dat.scl_inter = 0;
evc_vol = ismember(round(hcp_vol),EVC_ROI);
spm_write_vol(evc_hdr,evc_vol);
