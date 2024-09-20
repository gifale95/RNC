%% compare noise ceiling for with and without hrf fitting 

clear all
clc

%setup paths

fmri_path = '/data/pt_02350/derived/';
% set figurepath 
figure_path = '/data/pt_02348/objdraw/group_level/fmri/';

% add fmri path 

addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri')
% add libsvm

addpath(genpath('/data/pt_02348/objdraw/libsvm3.17'));

% add util function from meg folder 

addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/meg/utils');

%add tdt

addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'));

% get fmri subnames 

fmri_subs = dir(fullfile(fmri_path,'*sub*'));
fmri_subs = {fmri_subs.name}';

% specify excluded subjects
excluded_subjects = [7,9,10,13,22,23,29]; %od23because no mprage sequence there, 9, 10, 13 and 29 because bad data quality, 7,22 because missed a lot of catch trials 

% get config for experiment

cfg = config_subjects_objdraw();

% set correlation type 

corr_type = 'Pearson';

% initialize fMRI RDMs

EVC_fmri_photo_RDM = [];
LOC_fmri_photo_RDM = [];
EVC_fmri_photo_RDM_hrf_fitting = [];
LOC_fmri_photo_RDM_hrf_fitting = [];

res_name = 'RSA_final';

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    if any(ismember(excluded_subjects, cfg.subject_indices(sub_no))), continue, end 
    
    % load fMRI RDMs 
    
    load(fullfile(fmri_path,[sub_id], 'results','RSA_final','roi', 'Photo','res_other_average_RDV.mat'));
    
    EVC_fmri_photo_RDM = cat(3, EVC_fmri_photo_RDM, squareform(1-results.other_average_RDV.output{1}));
    LOC_fmri_photo_RDM = cat(3, LOC_fmri_photo_RDM, squareform(1-results.other_average_RDV.output{2}));
    
    load(fullfile(fmri_path,[sub_id], 'results','RSA_hrf_fitting','roi', 'Photo','res_other_average_RDV.mat'));
    
    EVC_fmri_photo_RDM_hrf_fitting = cat(3, EVC_fmri_photo_RDM_hrf_fitting, squareform(1-results.other_average_RDV.output{1}));
    LOC_fmri_photo_RDM_hrf_fitting = cat(3, LOC_fmri_photo_RDM_hrf_fitting, squareform(1-results.other_average_RDV.output{2}));
    
end 


    %% get noise ceiling using rsa toolbox 

addpath(genpath('/data/pt_02348/objdraw/fmri/rsatoolbox-1'))

[EVC_photo_upperBound, EVC_photo_lowerBound]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_photo_RDM,corr_type,1);
[LOC_photo_upperBound, LOC_photo_lowerBound]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_photo_RDM,corr_type,1);

[EVC_photo_upperBound_hrf_fitting, EVC_photo_lowerBound_hrf_fitting]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_photo_RDM_hrf_fitting,corr_type,1);
[LOC_photo_upperBound_hrf_fitting, LOC_photo_lowerBound_hrf_fitting]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_photo_RDM_hrf_fitting,corr_type,1);

%% plot easier 

figure
plot([EVC_photo_upperBound, EVC_photo_upperBound_hrf_fitting],'_','Markersize', 40, 'Color', rgb('Green'))
hold on 
plot([EVC_photo_lowerBound, EVC_photo_lowerBound_hrf_fitting],'_','Markersize', 40, 'Color', rgb('Black'))
xlim([0 3])
ylim([0 1])
xticks([0:3])
xticklabels({'';'No HRF-Fitting';'HRF-Fitting';''})
title('Noise Ceiling Estimates in EVC')
legend({'Upper Bound', 'Lower Bound'})

print(fullfile(figure_path, ['noise_ceiling_EVC_fmri_with_without_HRF_fitting.jpg']), ...
              '-djpeg', '-r300')
          
figure
plot([LOC_photo_upperBound,LOC_photo_upperBound_hrf_fitting],'_','Markersize', 40, 'Color', rgb('Green'))
hold on 
plot([LOC_photo_lowerBound, LOC_photo_lowerBound_hrf_fitting],'_','Markersize', 40, 'Color', rgb('Black'))
xlim([0 3])
ylim([0 1])
xticks([0:3])
xticklabels({'';'No HRF-Fitting';'HRF-Fitting';''})
title('Noise Ceiling Estimates in LOC')
legend({'Upper Bound', 'Lower Bound'})

print(fullfile(figure_path, ['noise_ceiling_LOC_fmri_with_without_HRF_fitting.jpg']), ...
              '-djpeg', '-r300')