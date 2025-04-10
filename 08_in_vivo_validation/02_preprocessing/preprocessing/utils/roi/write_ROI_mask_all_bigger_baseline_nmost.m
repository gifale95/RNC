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
function write_ROI_mask_all_bigger_baseline_nmost(cfg, i_sub,roi_name,nmost)

% load the contrast t-values
hdr = spm_vol(fullfile(cfg.sub(i_sub).dir, 'results','GLM','first_level_localizer_both_sessions','spmT_0003.nii'));
vol = spm_read_vols(hdr); 

% get the brain mask from the GLM
brainmask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir, 'results','GLM','hrf_fitting','fitted_explicit_brainmask','mask.nii')));

% load the subject specific EVC mask 

EVC_path = fullfile(cfg.sub(i_sub).dir,'roi',['r',roi_name,'_mask.nii']);
EVC_hdr = spm_vol(EVC_path);
EVC_vol = spm_read_vols(EVC_hdr);
fprintf('Size of the individual %s MNI image is %2f %2f %2f',roi_name,size(EVC_vol)); 

% find the n-most activated voxels  
vol = vol.*logical(EVC_vol).*logical(brainmask);

if sum(logical(vol),'all') < nmost 
    warning('The intersection between anatomical mask and localizer contrast does not contain enough voxels for the amount of selected most-activated voxels! Reconsider your specification!')
    mask_vol = logical(vol);
else
    if  sum(vol>0,'all') < nmost 
        maxkind = find(vol>0);
        randind = randsample(find(vol<0),nmost-length(maxkind)); % if there are not enough voxels with a positive value, then take a random sample
        maxkind = [maxkind;randind];
        mask_vol = zeros(size(vol));
        mask_vol(maxkind) = 1;
    else
        [~,maxkind] = maxk(abs(vol(:)),nmost);
        mask_vol = zeros(size(vol));
        mask_vol(maxkind) = 1;
    end
end
fprintf('\nNumber of most activated voxels %i\n', sum(mask_vol,'all')); 

maskhdr = spm_vol(EVC_path);
maskhdr.fname = fullfile(cfg.sub(i_sub).dir,'roi',[roi_name,'_mask_loc_nmost',num2str(nmost),'.nii']);
spm_write_vol(maskhdr,mask_vol);
%delete(EVC_path)
end 