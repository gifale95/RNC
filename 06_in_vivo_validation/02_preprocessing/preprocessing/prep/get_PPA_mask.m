%% script for writing MNI EVC mask from group level contrast

%setup some variables 

derived_dir = '/scratch/singej96/dfg_projekt/WP1/derived/exp/roi';

disp('Write PPA mask in MNI space')
% write mask for PPA
ppaL_hdr = spm_vol(fullfile(derived_dir,'scene_parcels',['lPPA.img']));
ppaL_vol = spm_read_vols(ppaL_hdr);
ppaR_hdr = spm_vol(fullfile(derived_dir,'scene_parcels',['rPPA.img']));
ppaR_vol = spm_read_vols(ppaR_hdr);

ppa_vol = ppaL_vol;
ppa_vol(ppaR_vol~=0) = 1; 

ppa_hdr = ppaL_hdr;
ppa_hdr.fname = fullfile(derived_dir,'ppa_mni.nii');
ppa_hdr.descrip = 'Kanwisher PPA defintion';
ppa_hdr.dt = [2 0];
ppa_hdr.private.dat.dtype = 'FLOAT32-LE';
ppa_hdr.private.dat.scl_slope = 1;
ppa_hdr.private.dat.scl_inter = 0;
spm_write_vol(ppa_hdr,ppa_vol);
