%% function for writing ROI mask for LO based on kanwisher parcels 

function write_ROI_mask_IT(cfg,IT_mask, i_sub,p,df)

% load the contrast t-values
hdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'results','GLM','first_level_denoise','spmT_0001.nii'));
vol = spm_read_vols(hdr);

% load LO mask to check for overlap 

LO_hdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'roi','LOmask_nklab.nii'));
LO_mask = spm_read_vols(LO_hdr);

% load IT mask transformed in individual subject space 

IT_hdr = spm_vol(IT_mask);
IT_vol = spm_read_vols(IT_hdr);

%check if p and df are given if not assign default value
if ~exist('p','var'), p=0.0001; end 
if ~exist('df','var'), df=294; end 

%threshold the t-values with a given p and df
T_thresh = tinv(1-p,df); % p , df repectively 

masked_vol = vol>T_thresh; 

IT_mask = masked_vol.*IT_vol; 
%IT_mask = IT_mask.*(~logical(LO_mask)); 
IT_hdr.fname = fullfile(cfg.sub(i_sub).dir,'roi','IT_mask.nii');
spm_write_vol(IT_hdr,IT_mask);
end 