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
function write_ROI_mask_EVC_nmost(cfg, i_sub,nmost)

% load the contrast t-values

localizer_dir = fullfile(cfg.sub(i_sub).dir, 'results','GLM','first_level_localizer_both_sessions');
hdr = spm_vol(fullfile(localizer_dir,'spmT_0003.nii'));

vol = spm_read_vols(hdr); 

% also load brainmask to be sure that there are really the N most
% activated voxels in the final mask 
brain_mask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir, 'results','GLM','hrf_fitting','fitted_explicit_brainmask','mask.nii')));

% load the subject specific EVC mask 

EVC_path = fullfile(cfg.sub(i_sub).dir,'roi','rtemp_evcmask.nii');
EVC_hdr = spm_vol(EVC_path);
EVC_vol = spm_read_vols(EVC_hdr);
fprintf('Size of the individual EVC MNI image is %2f %2f %2f',size(EVC_vol)); 

% find the n-most activated voxels  
vol = vol.*logical(EVC_vol).*brain_mask;
[~,maxkind] = maxk(vol(:),nmost); 
mask_vol = zeros(size(vol)); 
mask_vol(maxkind) = 1; 
fprintf('\nNumber of most activated voxels %i\n', sum(mask_vol,'all')); 

maskhdr = spm_vol(fullfile(localizer_dir,'mask.nii'));
maskhdr.fname = fullfile(cfg.sub(i_sub).dir,'roi',['evcmask_nmost',num2str(nmost),'.nii']);
spm_write_vol(maskhdr,mask_vol);
%delete(EVC_path)
end 