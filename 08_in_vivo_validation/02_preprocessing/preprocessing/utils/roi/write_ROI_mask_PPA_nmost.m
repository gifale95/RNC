% function  write_ROI_mask_PPA(cfg, i_sub, localizer_dir, p, df)
%
% This code can be used to compute the ROI mask for early visual cortex (PPA) which is defined 
% by taking the overlap between the subject specific PPA mask and significant cluster from the contrast of the localizer tak of the experiment.
% We do a number of things:
% 1. We get the contrast threshold the t-values with a p threshold (p)
% 2. We get the subject specific PPA mask 
% 3. We determine the overlap between the above threshold t-values and the
%    PPA mask and write the ROI mask 
%
%
function write_ROI_mask_PPA(cfg, i_sub, nmost)

% load the contrast t-values
localizer_dir = fullfile(cfg.sub(i_sub).dir, 'results','GLM','first_level_localizer_both_sessions');

hdr = spm_vol(fullfile(localizer_dir,'spmT_0002.nii'));
vol = spm_read_vols(hdr); 

% also load brainmask to be sure that there are really the N most
% activated voxels in the final mask 
brain_mask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir, 'results','GLM','hrf_fitting','fitted_explicit_brainmask','mask.nii')));

% load the subject specific PPA mask 

PPA_path = fullfile(cfg.sub(i_sub).dir,'roi','rtemp_PPA_mask.nii');
PPA_hdr = spm_vol(PPA_path);
PPA_vol = spm_read_vols(PPA_hdr);
fprintf('Size of the individual PPA MNI image is %2f %2f %2f',size(PPA_vol)); 

% find the n-most activated voxels  
vol = vol.*logical(PPA_vol).*brain_mask;
[~,maxkind] = maxk(vol(:),nmost); 
mask_vol = zeros(size(vol)); 
mask_vol(maxkind) = 1; 
fprintf('\nNumber of most activated voxels %i\n', sum(mask_vol,'all')); 

PPA_mask = mask_vol;
maskhdr = spm_vol(fullfile(localizer_dir,'mask.nii'));
maskhdr.fname = fullfile(cfg.sub(i_sub).dir,'roi',['PPA_mask_nmost',num2str(nmost),'.nii']);
spm_write_vol(maskhdr,PPA_mask);
%delete(PPA_path)
end 