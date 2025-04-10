%% script for writing the mask for all subjects from the normalized GLM results 
% just load all individual masks from the noramlized GLM results
% then get the overlap 
% then write the mask 


cfg = config_subjects_objdraw();

% add spm 
addpath('/data/pt_02348/objdraw/fmri/spm/spm12')

excluded_subjects = [7,9,10,13,22,23,29]; %od23because no mprage sequence there, 9, 10, 13 and 29 because bad data quality, 7,22 because missed a lot of catch trials 

all_mask = []; 

for i_sub = 1%:length(cfg.subject_indices) 
    
    if any(ismember(excluded_subjects, cfg.subject_indices(i_sub))), continue, end 
        
    hdr = spm_vol(fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir, 'results','GLM','hrf_fitting','mask_copy.nii'));
    vol = spm_read_vols(hdr);
    hdr.dt = [16 0]
    hdr.fname = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir, 'results','GLM','hrf_fitting','fitted','maskf16.nii')
    spm_write_vol(hdr,vol);

end 

% intersect_mask = (mean(all_mask,4)==1); 
% mask_hdr = spm_vol(fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir, 'results','GLM','hrf_fitting','normalized','wmask.nii'));
% mask_hdr.fname = fullfile(cfg.dirs.derived_dir, 'normalized_intersect_mask_hrf_fitting.nii'); 
% spm_write_vol(mask_hdr, intersect_mask); 