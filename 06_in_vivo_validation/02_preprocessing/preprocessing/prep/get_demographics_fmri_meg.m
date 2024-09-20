%% Objdraw - get the demographics of the participants 

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

% select modality -> fMRI and MEG main analysis used different sets of
% partly overlapping subjects because different subjects had to be excluded

modality = 'meg';

% get config for experiment 

cfg = config_subjects_objdraw();

switch modality
    case 'fmri'
        excluded_subjects = [7,9,10,13,22,23,29]; %od23because no mprage sequence there, 9, 10, 13 and 29 because bad data quality, 7,22 because missed a lot of catch trials 
    case 'meg'
        excluded_subjects = {'od01', 'od11','od17','od19','od23','od27','od28','od29'}; %od01,od17,od19,od28,od29 because below 80% in the paperclip taks, od11,od23,od27 because over 5% saccades with higher amplitude then 1.5°
end 

age = [];
gender = [];
% run loop over subject

for iSub = 1:30
    
    switch modality
        case 'fmri'
            if any(ismember(excluded_subjects, cfg.subject_indices(iSub))), continue, end
        case 'meg'
            if any(ismember(excluded_subjects, cfg.sub(cfg.subject_indices(iSub)).pid)), continue, end
    end
    age = cat(1,age,cfg.sub(iSub).age);
    gender = cat(1,gender,cfg.sub(iSub).gender);
    
    
end

fprintf('Mean age of subjects is %2f with standard deviation of %2f and %i females', mean(age), std(age), sum(gender)); 