%% roi control analysis 
% script to check if EVC and LOC+pfus overlap in any of the subjects 

clear all 
clc
cd '/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri'

% add all the helper functions 
addpath(genpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri'))
addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/meg/utils')

% specify path for the decoding toolbox here (which is needed for later
% steps in the analysis) 
addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'))

% path for tapas toolbox - for denoising 
addpath(genpath('/data/pt_02348/objdraw/fmri/tapas-master'))

% dir for saving data diagnostics plots 

sancheck_figures = '/data/pt_02350/group/datadiagnostics';

% get config for experiment 

cfg = config_subjects_objdraw();

excluded_subjects = [7,9,10,13,22,23,29]; 

for sub = 1:30 
    
    if any(ismember(excluded_subjects, cfg.subject_indices(sub))), continue, end
    
    isub = cfg.subject_indices(sub); 
    
    roi_dir = fullfile(cfg.sub(isub).dir,'roi');
    
    masks = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'combined_loc_fus_mask.nii')};
    
    mask1 = spm_read_vols(spm_vol(masks{1}));
    mask2 = spm_read_vols(spm_vol(masks{2})); 
    
    overlap{sub} = intersect(find(mask1==1), find(mask2==1)); 
    
    if ~isempty(overlap{sub}) 
        fprintf('\nEVC and LOC ROIs overlap in subject %s', cfg.sub(isub).pid)
    end 
end 

%% remove overlap 

for sub = 1:30 
        
    if any(ismember(excluded_subjects, cfg.subject_indices(sub))), continue, end
    
    isub = cfg.subject_indices(sub); 

    check_ROI_overlap(cfg,isub); 
end 