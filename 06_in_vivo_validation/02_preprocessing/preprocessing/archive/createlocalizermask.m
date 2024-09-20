%% function for creating localizer mask 

function createlocalizermask(fname, outpath, thr, df)


% load the contrast t-values
hdr = spm_vol(fname);
vol = spm_read_vols(hdr);  

%check if p and df are given if not assign default value
if ~exist('thr','var'), p=0.0001; end 
if ~exist('df','var'), df=294; end 

%threshold the t-values with a given p and df
T_thresh = tinv(1-p,df); % p , df repectively 

masked_vol = vol>T_thresh; 

locmask = masked_vol;
maskhdr = hdr;
maskhdr.fname = fullfile(outpath,'locT_mask.nii');
spm_write_vol(maskhdr,locmask);

end 