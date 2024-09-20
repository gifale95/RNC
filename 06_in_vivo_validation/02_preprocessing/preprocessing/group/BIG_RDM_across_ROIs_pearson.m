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

% initialize fMRI RDMs

EVC_fmri_BIG_RDM = [];
LOC_fmri_BIG_RDM = [];

res_name = 'BIG_RSA_denoise_noisenorm';

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    if any(ismember(excluded_subjects, cfg.subject_indices(sub_no))), continue, end 
    
    % load fMRI RDMs 
    
    load(fullfile(fmri_path,[sub_id], 'results',res_name,'roi','all','res_other_average_RDV.mat'));
    
    EVC_fmri_BIG_RDM = cat(2, EVC_fmri_BIG_RDM, 1-results.other_average_RDV.output{1});
    LOC_fmri_BIG_RDM = cat(2, LOC_fmri_BIG_RDM, 1-results.other_average_RDV.output{2});

    
end


%% plot RDMs with sorting 

%load the sort vector 

load('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri/utils/sort_vector.mat')

% compute the mean for every ROI and depiction 

mean_fmri_RDM(:,:,1) = squareform(mean(EVC_fmri_BIG_RDM,2));
mean_fmri_RDM(:,:,2) = squareform(mean(LOC_fmri_BIG_RDM,2));


% sort the RDMs 
[~ ,sort_idxs] = sort(sel_sort_vector);
n_cat = size(mean_fmri_RDM,1)/3;
sort_idxs =[sort_idxs sort_idxs+n_cat sort_idxs+n_cat*2];
mean_fmri_RDM = mean_fmri_RDM(sort_idxs, sort_idxs,:,:);

% plot the RDMs
cmap = colormap('inferno');

roi_names = {'EVC'; 'LO'};
figure('units','normalized','outerposition',[0 0 1 1])
for roi = 1:2 
        
        subplot(1,2,roi)
        
        imagesc(squareform(tiedrank(squareform(mean_fmri_RDM(:,:,roi))))); 
        colormap(cmap);
        title(['BIG RDM ',roi_names{roi}])
        set(gca,'xtick',[])
        set(gca,'xticklabel',[])
        set(gca,'ytick',[])
        set(gca,'yticklabel',[])
        axis square
end

print(fullfile(figure_path, ['sorted_BIG_RDMs_ROI_pearson.jpg']), ...
              '-djpeg', '-r300')

%% perform MDS on the mean fmri RDMs

for roi = 1:2 
    
    BIG_MDS{roi} = mdscale(mean_fmri_RDM(:,:,roi),2);
    
end 

%% plot MDS 


figure
for roi=1:length(BIG_MDS)
subplot(1,2,roi)
scatter(BIG_MDS{roi}(1:n_cat,1),BIG_MDS{roi}(1:n_cat,2),'filled', 'black','LineWidth', 6);
hold on 
scatter(BIG_MDS{roi}(n_cat+1:n_cat*2,1),BIG_MDS{roi}(n_cat+1:n_cat*2,2),'filled', 'MarkerFaceColor',cmap(ceil(256*0.3),:),'LineWidth', 6);
scatter(BIG_MDS{roi}(n_cat*2+1:n_cat*3,1),BIG_MDS{roi}(n_cat*2+1:n_cat*3,2),'filled', 'MarkerFaceColor', cmap(ceil(256*0.6),:),'LineWidth', 6);
title(['MDS for ' roi_names{roi}])
end 
suptitle(['MDS for all depictions together across rois'])
hL = legend('Photos', 'Drawings', 'Sketches');
newPosition = [0.85 0.4 0.2 0.2];
newUnits = 'normalized';
set(hL,'Position', newPosition,'Units', newUnits);

print(fullfile(figure_path, ['BIG_MDS_roi_pearson.jpg']), ...
              '-djpeg', '-r300')

%% get noise ceiling using rsa toolbox 

addpath(genpath('/data/pt_02348/objdraw/fmri/rsatoolbox-1'))

[EVC_photo_upperBound, EVC_photo_lowerBound]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_photo_RDM,'Spearman',1);
[LO_photo_upperBound, LO_photo_lowerBound]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_BIG_RDM,'Spearman',1);

[EVC_drawing_upperBound, EVC_drawing_lowerBound]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_drawing_RDM,'Spearman',1);
[LO_drawing_upperBound, LO_drawing_lowerBound]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_drawing_RDM,'Spearman',1);
  
[EVC_sketch_upperBound, EVC_sketch_lowerBound]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_sketch_RDM,'Spearman',1);
[LO_sketch_upperBound, LO_sketch_lowerBound]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_sketch_RDM, 'Spearman',1); 


  