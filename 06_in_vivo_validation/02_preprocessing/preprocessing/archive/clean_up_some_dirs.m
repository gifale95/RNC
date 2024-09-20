%% fMRI RSA-ROI analysis wrapper

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


for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    %if any(ismember(excluded_subjects, cfg.subject_indices(sub_no))), continue, end 
    
    % load fMRI RDMs 
    
   if isdir(fullfile(fmri_path,[sub_id], 'results','GLM','GLMsingle'))
       rmdir(fullfile(fmri_path,[sub_id], 'results','GLM','GLMsingle'),'s')
   end 
   
   if isdir(fullfile(fmri_path,[sub_id], 'results','GLM','first_level_GLMdenoise'))
       rmdir(fullfile(fmri_path,[sub_id], 'results','GLM','first_level_GLMdenoise'),'s')
   end
   
      if isdir(fullfile(fmri_path,[sub_id], 'results','RSA_crossnobis_normalized'))
       rmdir(fullfile(fmri_path,[sub_id], 'results','RSA_crossnobis_normalized'),'s')
      end
          if isdir(fullfile(fmri_path,[sub_id], 'results','RSA_denoise_IT'))
       rmdir(fullfile(fmri_path,[sub_id], 'results','RSA_denoise_IT'),'s')
          end
   
             if isdir(fullfile(fmri_path,[sub_id], 'results','crossdecoding','cat_noisenorm_samevoxsz'))
       rmdir(fullfile(fmri_path,[sub_id], 'results','crossdecoding','cat_noisenorm_samevoxsz'),'s')
          end
          
end
