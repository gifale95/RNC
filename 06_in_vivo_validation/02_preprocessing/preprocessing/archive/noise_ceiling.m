%% fMRI RSA-ROI noiseceiling searchlight

clear all
clc

%setup paths

fmri_path = '/data/pt_02350/derived/';

% add libsvm

addpath(genpath('/data/pt_02348/objdraw/libsvm3.17'));

% add util function from meg folder 

addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/meg');

% get fmri subnames 

fmri_subs = dir(fullfile(fmri_path,'*sub*'));
fmri_subs = {fmri_subs.name}';

% specify excluded subjects

excluded_subs = {'sub07','sub12','sub13','sub22','sub23','sub29','sub31'};

% initialize fMRI RDMs

fmri_RDM = [];

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    if any(ismember(excluded_subs,sub_id)), continue, end,
    
    % load fMRI RDMs 
    
    load(fullfile(fmri_path,[sub_id], 'results','RSA_denoise','searchlight', 'all','res_other.mat'));
    
    this_RDM = results.other.output(:);
    fmri_RDM(:,:,:,sub_no) = 
   

end

%% compute noise ceiling for both ROIS

addpath(genpath('/data/pt_02348/objdraw/fmri/rsatoolbox-1'))

[EVC_ceiling_upperBound, EVC_ceiling_lowerBound, EVC_bestFitRDM]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_RDM,'Spearman',1);
[LOC_ceiling_upperBound, LOC_ceiling_lowerBound, LOC_bestFitRDM]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_RDM,'Spearman',1);



