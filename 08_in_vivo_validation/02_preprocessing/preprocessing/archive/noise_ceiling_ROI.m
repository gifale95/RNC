%% fMRI RSA-ROI noiseceiling

clear all
clc

%setup paths

fmri_path = '/data/pt_02350/derived/';
figure_path = '/data/pt_02348/objdraw/group_level/fmri/';

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

EVC_fmri_RDM = [];
LOC_fmri_RDM = [];

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    if any(ismember(excluded_subs,sub_id)), continue, end,
    
    % load fMRI RDMs 
    
    load(fullfile(fmri_path,[sub_id], 'results','RSA_denoise','roi', 'all','res_other.mat'));
    
    EVC_fmri_RDM = cat(3, EVC_fmri_RDM, 1-results.other.output{1});
    LOC_fmri_RDM = cat(3, LOC_fmri_RDM, 1-results.other.output{2});
   

end

%% compute noise ceiling for both ROIS

addpath(genpath('/data/pt_02348/objdraw/fmri/rsatoolbox-1'))

[EVC_ceiling_upperBound, EVC_ceiling_lowerBound, EVC_bestFitRDM]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_RDM,'Spearman',1);
[LOC_ceiling_upperBound, LOC_ceiling_lowerBound, LOC_bestFitRDM]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_RDM,'Spearman',1);

%% plot 

y = [mean([EVC_ceiling_lowerBound, EVC_ceiling_upperBound]), mean([LOC_ceiling_lowerBound,LOC_ceiling_upperBound])];
ceiling_upper = [EVC_ceiling_upperBound-y(1), LOC_ceiling_upperBound-y(2)];
ceiling_lower = [abs(EVC_ceiling_lowerBound-y(1)), abs(LOC_ceiling_lowerBound-y(2))];
errorbar([1:2],y,ceiling_lower, ceiling_upper)
xlim([0 3])
ylim([0 1])

%% plot easier 

plot([EVC_ceiling_upperBound, LOC_ceiling_upperBound],'_','Markersize', 40, 'Color', rgb('Green'))
hold on 
plot([EVC_ceiling_lowerBound, LOC_ceiling_lowerBound],'_','Markersize', 40, 'Color', rgb('Black'))
xlim([0 3])
ylim([0 1])
xticks([0:3])
xticklabels({'';'EVC';'LOC';''})
title('Noise Ceiling Estimates across ROIs')
legend({'Upper Bound', 'Lower Bound'})

print(fullfile(figure_path, ['noise_ceiling_ROI_fmri.jpg']), ...
              '-djpeg', '-r300')