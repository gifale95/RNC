%% function for writing ROI mask for LO based on kanwisher parcels 

function write_ROI_mask_IT_nklab(cfg,LO_mask, i_sub,p,df)

% load the contrast t-values
hdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'results','GLM','localizer','spmT_0001.nii'));
vol = spm_read_vols(hdr);

% load nklab parcel transformed in individual subject space 

LO_hdr = spm_vol(LO_mask);
LO_vol = spm_read_vols(LO_hdr);

%check if p and df are given if not assign default value
if ~exist('p','var'), p=0.0001, end 
if ~exist('df','var'), df=294, end 

%threshold the t-values with a given p and df
T_thresh = tinv(1-p,df); % p , df repectively 

masked_vol = vol>T_thresh; 

LO_mask = masked_vol.*LO_vol; 
LO_hdr.fname = fullfile(cfg.sub(i_sub).dir,'roi','LOmask_nklab.nii');
spm_write_vol(LO_hdr,LO_mask);
end 