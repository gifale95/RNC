% function  write_ROI_mask_EVC(cfg, i_sub, localizer_dir, p, df)
%
% This code can be used to compute the ROI mask for early visual cortex (EVC) which is defined 
% by taking n-most activated voxels for the localizer contrast in a given subject specific EVC mask. 
% We do a number of things:
% 1. We get the contrast 
% 2. We get the subject specific EVC mask 
% 3. We determine n-most activated voxels in the
%    EVC mask and write the ROI mask 
%
%
function write_ROI_mask_full_anatomical(cfg, i_sub,roi_name); 

% get the brain mask from the GLM
brainmask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir, 'results','GLM','hrf_fitting','fitted_explicit_brainmask','mask.nii')));

% load the subject specific EVC mask 

EVC_path = fullfile(cfg.sub(i_sub).dir,'roi',['r',roi_name,'_mask.nii']);
EVC_hdr = spm_vol(EVC_path);
EVC_vol = spm_read_vols(EVC_hdr);
fprintf('Size of the individual %s MNI image is %2f %2f %2f',roi_name,size(EVC_vol)); 

% find the voxels that lie inside the brain mask
vol = EVC_vol.*logical(brainmask);

% take all voxels that have values 
mask_vol = logical(vol);
fprintf('\nNumber of voxels in ROI %i\n', sum(mask_vol,'all')); 

maskhdr = spm_vol(EVC_path);
maskhdr.fname = fullfile(cfg.sub(i_sub).dir,'roi',[roi_name,'_mask_full_anatomical.nii']);
spm_write_vol(maskhdr,mask_vol);
%delete(EVC_path)
end 