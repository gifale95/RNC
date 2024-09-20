% function  write_ROI_mask_EVC(cfg, i_sub)
%
% This code can be used to compute the ROI mask for early visual cortex (EVC) which is defined 
% by taking the overlap between the subject specific EVC mask and significant cluster from the contrast of the localizer tak of the experiment.
% We do a number of things:
% 1. We get the contrast threshold the t-values with a p threshold (p)
% 2. We get the subject specific EVC mask 
% 3. We determine the overlap between the above threshold t-values and the
%    EVC mask and write the ROI mask 
%
%
function write_ROI_mask(cfg, i_sub,mask_name)

% load the contrast t-values
hdr = spm_vol(fullfile(cfg.sub(i_sub).dir, 'results','GLM','first_level','spmT_0001.nii'));
vol = spm_read_vols(hdr); 

% load the subject specific mask 
mask_hdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['r',mask_name]));
mask_vol = spm_read_vols(mask_hdr);
fprintf('Size of the individual mask image is %2f %2f %2f',size(mask_vol)); 

% mask the t-map with the ROI mask 
masked_vol = vol.*mask_vol; 

% get 10% percentile for the contrast 
perc = prctile(vol(logical(mask_vol)),90, 'all');
most_act_vol = masked_vol > perc; 
fprintf('\nNumber of most activated voxels %i\n', sum(most_act_vol,'all')); 

maskhdr = spm_vol(fullfile(cfg.sub(i_sub).dir, 'results','GLM','first_level','mask.nii'));
maskhdr.fname = fullfile(cfg.sub(i_sub).dir,'roi',mask_name); 
spm_write_vol(maskhdr,most_act_vol);

end 