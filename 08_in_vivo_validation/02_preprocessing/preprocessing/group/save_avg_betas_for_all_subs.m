%% get ROI betas from beta avgs for tuned RSA

clear all 
clc

addpath(genpath('/data/pt_02348/objdraw/fmri/martin_spm'))
addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'))
addpath(genpath('/data/hu_jsinger/Documents/MATLAB/natsort/'))

% setup constants 

n_cat = 48; % number of object categories 

% get config for experiment 

cfg = config_subjects_objdraw();

excluded_subjects = [7,12,13,22,23,29,31]; %od23 because no mprage sequence there, 12 and 29 because bad data quality

for iSub = 1:30
    
    if any(ismember(excluded_subjects, cfg.subject_indices(iSub))), continue, end 
    
    avg_dir = fullfile(cfg.sub(cfg.subject_indices(iSub)).dir,'results','GLM','first_level_denoise', 'avg');
    roi_dir = fullfile(cfg.sub(cfg.subject_indices(iSub)).dir,'roi');
    
    % load ROI masks
    EVC_mask = spm_read_vols(spm_vol(fullfile(roi_dir,'evcmask.nii')));
    LO_mask = spm_read_vols(spm_vol(fullfile(roi_dir,'combined_loc_fus_mask.nii')));
    
    avg_files = dir(fullfile(avg_dir, '*.nii')); 
    avg_filenames = {avg_files.name}';
    avg_filenames = natsortfiles(avg_filenames);
    
    these_EVC_betas=[];
    these_LO_betas = [];
    
    for i = 1:length(avg_files) 
        
        beta = spm_read_vols(spm_vol(fullfile(avg_dir,avg_filenames{i})));
        these_EVC_betas = cat(1,these_EVC_betas, reshape(beta(logical(EVC_mask)),1,[]));
        these_LO_betas = cat(1,these_LO_betas, reshape(beta(logical(LO_mask)),1,[]));
    end 
    
    EVC_photo_betas = these_EVC_betas(49:96,:);
    save(fullfile(cfg.sub(cfg.subject_indices(iSub)).dir,'results','GLM','first_level_denoise', 'avg', 'all_photo_EVC_betas.mat'),'EVC_photo_betas');
    EVC_drawing_betas = these_EVC_betas(1:48,:);
    save(fullfile(cfg.sub(cfg.subject_indices(iSub)).dir,'results','GLM','first_level_denoise', 'avg', 'all_drawing_EVC_betas.mat'),'EVC_drawing_betas');
    EVC_sketch_betas = these_EVC_betas(97:end,:);
    save(fullfile(cfg.sub(cfg.subject_indices(iSub)).dir,'results','GLM','first_level_denoise', 'avg', 'all_sketch_EVC_betas.mat'),'EVC_sketch_betas');

    LO_photo_betas = these_LO_betas(49:96,:);
    save(fullfile(cfg.sub(cfg.subject_indices(iSub)).dir,'results','GLM','first_level_denoise', 'avg', 'all_photo_LOC_betas.mat'),'LO_photo_betas');
    LO_drawing_betas = these_LO_betas(1:48,:);
    save(fullfile(cfg.sub(cfg.subject_indices(iSub)).dir,'results','GLM','first_level_denoise', 'avg', 'all_drawing_LOC_betas.mat'),'LO_drawing_betas');
    LO_sketch_betas = these_LO_betas(97:end,:);
    save(fullfile(cfg.sub(cfg.subject_indices(iSub)).dir,'results','GLM','first_level_denoise', 'avg', 'all_sketch_LOC_betas.mat'),'LO_sketch_betas');

    fprintf('Subject: %f EVC mask size is %f and LO mask size is %f\n', cfg.subject_indices(iSub), size(EVC_photo_betas,2),size(LO_photo_betas,2))  
    
end 