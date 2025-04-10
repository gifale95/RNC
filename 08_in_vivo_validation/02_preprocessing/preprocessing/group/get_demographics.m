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

% get config for experiment 

cfg = config_subjects_objdraw();

excluded_subjects = [7,9,10,13,22,23,29]; %od23because no mprage sequence there, 9, 10, 13 and 29 because bad data quality, 7,22 because missed a lot of catch trials 

age = [];
gender = [];
% run loop over subject 

for iSub = 1:30
   
    if any(ismember(excluded_subjects, cfg.subject_indices(iSub))), continue, end
    
    age = cat(1,age,cfg.sub(iSub).age);
    gender = cat(1,gender,cfg.sub(iSub).gender); 
    
   
end 