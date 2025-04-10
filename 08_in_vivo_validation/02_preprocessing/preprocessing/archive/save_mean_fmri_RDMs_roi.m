%% script for computing the average RDMs for both ROIs (EVC,LOC) in preparation for the FRRSA fitting
% just load all individual roi RSA results and average them 
% subsequently save the average RDMs 

clear all
clc

%setup paths

fmri_path = '/data/pt_02350/derived/';

save_path = '/data/pt_02350/group/mean_RDMs/';

% get fmri subnames 

fmri_subs = dir(fullfile(fmri_path,'*sub*'));
fmri_subs = {fmri_subs.name}';

% specify excluded subjects
excluded_subjects = [7,9,10,13,22,23,29]; %od23 because no mprage sequence there, 9,10,13 and 29 because bad data quality 7,22 because below 80% in paperclip task

% get config for experiment

cfg = config_subjects_objdraw();
% initialize fMRI RDMs

EVC_fmri_photo_RDM = [];
LOC_fmri_photo_RDM = [];
EVC_fmri_drawing_RDM = [];
LOC_fmri_drawing_RDM = [];
EVC_fmri_sketch_RDM = [];
LOC_fmri_sketch_RDM = [];

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    if any(ismember(excluded_subjects, cfg.subject_indices(sub_no))), continue, end 
    
    % load fMRI RDMs 
    
    load(fullfile(fmri_path,[sub_id], 'results','RSA_noisenorm_pearson','roi', 'Photo','res_other_average_RDV.mat'));
    
    EVC_fmri_photo_RDM = cat(2, EVC_fmri_photo_RDM, results.other_average_RDV.output{1});
    LOC_fmri_photo_RDM = cat(2, LOC_fmri_photo_RDM, results.other_average_RDV.output{2});
    
    load(fullfile(fmri_path,sub_id, 'results','RSA_noisenorm_pearson','roi', 'Drawing','res_other_average_RDV.mat'));
    
    EVC_fmri_drawing_RDM = cat(2, EVC_fmri_drawing_RDM, results.other_average_RDV.output{1});
    LOC_fmri_drawing_RDM = cat(2, LOC_fmri_drawing_RDM, results.other_average_RDV.output{2});
    
    load(fullfile(fmri_path,sub_id, 'results','RSA_noisenorm_pearson','roi', 'Sketch','res_other_average_RDV.mat'));
    
    EVC_fmri_sketch_RDM = cat(2, EVC_fmri_sketch_RDM, results.other_average_RDV.output{1});
    LOC_fmri_sketch_RDM = cat(2, LOC_fmri_sketch_RDM, results.other_average_RDV.output{2});

    
end

%% average and save results 

mean_EVC_photo_RDM = 1-squareform(mean(EVC_fmri_photo_RDM,2));
mean_EVC_drawing_RDM = 1-squareform(mean(EVC_fmri_drawing_RDM,2));
mean_EVC_sketch_RDM = 1-squareform(mean(EVC_fmri_sketch_RDM,2));

mean_LOC_photo_RDM = 1-squareform(mean(LOC_fmri_photo_RDM,2));
mean_LOC_drawing_RDM = 1-squareform(mean(LOC_fmri_drawing_RDM,2));
mean_LOC_sketch_RDM = 1-squareform(mean(LOC_fmri_sketch_RDM,2));

save(fullfile(save_path, 'mean_EVC_photo_RDM.mat'), 'mean_EVC_photo_RDM')
save(fullfile(save_path, 'mean_EVC_drawing_RDM.mat'), 'mean_EVC_drawing_RDM')
save(fullfile(save_path, 'mean_EVC_sketch_RDM.mat'), 'mean_EVC_sketch_RDM')
save(fullfile(save_path, 'mean_LOC_photo_RDM.mat'), 'mean_LOC_photo_RDM')
save(fullfile(save_path, 'mean_LOC_drawing_RDM.mat'), 'mean_LOC_drawing_RDM')
save(fullfile(save_path, 'mean_LOC_sketch_RDM.mat'), 'mean_LOC_sketch_RDM')