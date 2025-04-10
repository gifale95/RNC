% function  write_ROI_mask_loc(cfg, i_sub, localizer_dir, p, df)
%
% This code can be used to compute the ROI mask for LOC which is defined 
% by taking the overlap between the subject specific LOC mask and significant cluster from the contrast of the localizer tak of the experiment.
% We do a number of things:
% 1. We get the contrast threshold the t-values with a p threshold (p)
% 2. We get the subject specific LOC mask 
% 3. We determine the overlap between the above threshold t-values and the
%    LOC mask and write the ROI mask 
%
%
function write_ROI_mask_LOC(cfg, i_sub, p, df)

% load the contrast t-values

localizer_dir = fullfile(cfg.sub(i_sub).dir, 'results','GLM','first_level_localizer_both_sessions');

hdr = spm_vol(fullfile(localizer_dir,'spmT_0001.nii'));
vol = spm_read_vols(hdr); 

% load the subject specific LOC mask 

loc_path = fullfile(cfg.sub(i_sub).dir,'roi','rloc_mask.nii');
loc_hdr = spm_vol(loc_path);
loc_vol = spm_read_vols(loc_hdr);
fprintf('Size of the individual LOC image is %2f %2f %2f',size(loc_vol)); 

%check if p and df are given if not assign default value
if ~exist('p','var'), p=0.0001; end 
if ~exist('df','var')
    if i_sub <4
       df=427; 
    else
        df=854;
    end
end
%threshold the t-values with a given p and df
T_thresh = tinv(1-p,df); % p , df repectively 

masked_vol = vol>T_thresh; 

loc_mask = loc_vol.*masked_vol;
maskhdr = spm_vol(fullfile(localizer_dir,'mask.nii'));
maskhdr.fname = fullfile(cfg.sub(i_sub).dir,'roi','loc_mask.nii');
spm_write_vol(maskhdr,loc_mask);
delete(loc_path)
end 