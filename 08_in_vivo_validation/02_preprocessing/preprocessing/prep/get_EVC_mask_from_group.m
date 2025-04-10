%% script for writing MNI EVC mask from group level contrast

%setup some variables 

derived_dir = '/data/pt_02350/derived/';

disp('Write EVC mask in MNI space')
% write mask for V1-V3 (still in MNI space)
EVC_ROI = [1, 4, 5];
hcp_hdr = spm_vol(fullfile(derived_dir,['HCP-MMP1_on_MNI152_ICBM2009b_nlin_asym.nii']));
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

% resample the EVC MNI mask to the spm MNI space 
spm('defaults','fmri')
matlabbatch{1}.spm.spatial.coreg.write.ref = {'/data/pt_02350/group/EVC_ROI/spmT_0001.nii'};
matlabbatch{1}.spm.spatial.coreg.write.source = {fullfile(derived_dir, 'evc_mni.nii,1')};
matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 1;
matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'spm_';

disp('Resampling EVC mask from atlas MNI space into SPM MNI space')
spm_jobman('run',matlabbatch)

% get the overlap between the group contrast and the EVC mask in spm MNI
% space

% first load the contrast image 
group_con_path = '/data/pt_02350/group/EVC_ROI/spmT_0001.nii';
group_con_hdr = spm_vol(group_con_path);
group_con_vol = spm_read_vols(group_con_hdr);

% then load the EVC image in SPM MNI space
EVC_path = fullfile(derived_dir, 'spm_evc_mni.nii');
EVC_hdr = spm_vol(EVC_path);
EVC_vol = spm_read_vols(EVC_hdr);

% threshold the t-contrast 
p=0.001;
df=26;

%threshold the t-values with a given p and df
T_thresh = tinv(1-p,df); % p , df repectively 

masked_con_vol = group_con_vol>T_thresh; 

% get the overlap between the EVC mask and the t-contrast 

new_EVC_mask = masked_con_vol.*EVC_vol; 

% get a hdr for the new EVC mas, assign new name and write volume 

EVC_mask_hdr = EVC_hdr; 
EVC_mask_hdr.fname = fullfile(derived_dir, 'mni_evcmask.nii');
spm_write_vol(EVC_mask_hdr,new_EVC_mask);
