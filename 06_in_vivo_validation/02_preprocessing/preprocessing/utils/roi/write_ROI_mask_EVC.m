% function  write_ROI_mask_EVC(cfg, i_sub, localizer_dir, p, df)
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
function write_ROI_mask_EVC(cfg, i_sub, p, df)


localizer_dir = fullfile(cfg.sub(i_sub).dir, 'results','GLM','first_level_localizer_both_sessions');

hdr = spm_vol(fullfile(localizer_dir,'spmT_0003.nii'));

vol = spm_read_vols(hdr); 

% load the subject specific EVC mask 

EVC_path = fullfile(cfg.sub(i_sub).dir,'roi','revcmask.nii');
EVC_hdr = spm_vol(EVC_path);
EVC_vol = spm_read_vols(EVC_hdr);
fprintf('Size of the individual EVC MNI image is %2f %2f %2f',size(EVC_vol)); 

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

EVC_mask = EVC_vol.*masked_vol;
maskhdr = spm_vol(fullfile(localizer_dir,'mask.nii'));
maskhdr.fname = fullfile(cfg.sub(i_sub).dir,'roi','evcmask.nii');
spm_write_vol(maskhdr,EVC_mask);
delete(EVC_path)
end 