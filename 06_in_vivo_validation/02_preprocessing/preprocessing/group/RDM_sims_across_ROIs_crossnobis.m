%% fMRI RSA-ROI analysis wrapper

clear all
clc

%setup paths

fmri_path = '/data/pt_02350/derived/';
% set figurepath 
figure_path = '/data/pt_02348/objdraw/group_level/fmri/';


% add libsvm

addpath(genpath('/data/pt_02348/objdraw/libsvm3.17'));

% add util function from meg folder 

addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/meg');

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

EVC_fmri_photo_RDM = [];
LOC_fmri_photo_RDM = [];
pfus_fmri_photo_RDM = [];
EVC_fmri_drawing_RDM = [];
LOC_fmri_drawing_RDM = [];
pfus_fmri_drawing_RDM = [];
EVC_fmri_sketch_RDM = [];
LOC_fmri_sketch_RDM = [];
pfus_fmri_sketch_RDM = [];

res_name = 'RSA_noisenorm_pearson';

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    if any(ismember(excluded_subjects, cfg.subject_indices(sub_no))), continue, end 
    
    % load fMRI RDMs 
    
    load(fullfile(fmri_path,[sub_id], 'results',res_name,'roi', 'Photo','res_other_average_RDV.mat'));
    
    EVC_fmri_photo_RDM = cat(2, EVC_fmri_photo_RDM, results.other_average_RDV.output{1});
    LOC_fmri_photo_RDM = cat(2, LOC_fmri_photo_RDM, results.other_average_RDV.output{2});
    
    load(fullfile(fmri_path,sub_id, 'results',res_name,'roi', 'Drawing','res_other_average_RDV.mat'));
    
    EVC_fmri_drawing_RDM = cat(2, EVC_fmri_drawing_RDM, results.other_average_RDV.output{1});
    LOC_fmri_drawing_RDM = cat(2, LOC_fmri_drawing_RDM, results.other_average_RDV.output{2});
    
    load(fullfile(fmri_path,sub_id, 'results',res_name,'roi', 'Sketch','res_other_average_RDV.mat'));
    
    EVC_fmri_sketch_RDM = cat(2, EVC_fmri_sketch_RDM, results.other_average_RDV.output{1});
    LOC_fmri_sketch_RDM = cat(2, LOC_fmri_sketch_RDM, results.other_average_RDV.output{2});

    
end


%% plot RDMs with sorting 

%load the sort vector 

load('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri/utils/sort_vector.mat')

% compute the mean for every ROI and depiction 

mean_fmri_RDM(:,:,1,1) = squareform(mean(1-EVC_fmri_photo_RDM,2));
mean_fmri_RDM(:,:,1,2) = squareform(mean(1-EVC_fmri_drawing_RDM,2));
mean_fmri_RDM(:,:,1,3) = squareform(mean(1-EVC_fmri_sketch_RDM,2));

mean_fmri_RDM(:,:,2,1) = squareform(mean(1-LOC_fmri_photo_RDM,2));
mean_fmri_RDM(:,:,2,2) = squareform(mean(1-LOC_fmri_drawing_RDM,2));
mean_fmri_RDM(:,:,2,3) = squareform(mean(1-LOC_fmri_sketch_RDM,2));

% sort the RDMs 
[~ ,sort_idxs] = sort(sel_sort_vector);
mean_fmri_RDM = mean_fmri_RDM(sort_idxs, sort_idxs,:,:);

% plot the RDMs
cmap = colormap('inferno');
ct = 1;
for roi = 1:2 
    
    for cond = 1:3
        
        
        subplot(2,3,ct)
        
        imagesc(squareform(tiedrank(1-squareform(mean_fmri_RDM(:,:,roi,cond))))); 
        colormap(cmap);
        ct= ct+1;
    end
end

%% compute RDM similarity 

for sub = 1:size(EVC_fmri_photo_RDM,3)
    
    EVC_photo_drawing(sub) = corr(squareformq(EVC_fmri_photo_RDM(:,:,sub)),squareformq(EVC_fmri_drawing_RDM(:,:,sub)),'Type','Spearman');
    EVC_photo_sketch(sub) = corr(squareformq(EVC_fmri_photo_RDM(:,:,sub)),squareformq(EVC_fmri_sketch_RDM(:,:,sub)),'Type','Spearman');
    EVC_drawing_sketch(sub) = corr(squareformq(EVC_fmri_drawing_RDM(:,:,sub)),squareformq(EVC_fmri_sketch_RDM(:,:,sub)),'Type','Spearman');

    LOC_photo_drawing(sub) = corr(squareformq(LOC_fmri_photo_RDM(:,:,sub)),squareformq(LOC_fmri_drawing_RDM(:,:,sub)),'Type','Spearman');
    LOC_photo_sketch(sub) = corr(squareformq(LOC_fmri_photo_RDM(:,:,sub)),squareformq(LOC_fmri_sketch_RDM(:,:,sub)),'Type','Spearman');
    LOC_drawing_sketch(sub) = corr(squareformq(LOC_fmri_drawing_RDM(:,:,sub)),squareformq(LOC_fmri_sketch_RDM(:,:,sub)),'Type','Spearman');

end 

%% plot RDM similarities 

cmap = colormap('inferno');

roi_names = {'EVC'; 'LO'};

all_sims = cat(1, [mean(EVC_photo_drawing),mean(EVC_photo_sketch),mean(EVC_drawing_sketch)],...
                  [mean(LOC_photo_drawing),mean(LOC_photo_sketch),mean(LOC_drawing_sketch)]);
EVC_se = [std(EVC_photo_drawing)/sqrt(length(EVC_photo_drawing)),...
            std(EVC_photo_sketch)/sqrt(length(EVC_photo_sketch)),...
            std(EVC_drawing_sketch)/sqrt(length(EVC_drawing_sketch))];
LOC_se = [std(LOC_photo_drawing)/sqrt(length(LOC_photo_drawing)),...
            std(LOC_photo_sketch)/sqrt(length(LOC_photo_sketch)),...
            std(LOC_drawing_sketch)/sqrt(length(LOC_drawing_sketch))];
        
all_se = cat(1, EVC_se, LOC_se);

figure
h = bar(all_sims, 'grouped', 'FaceColor', 'flat');
h(1,1).CData(1,:) = rgb('Black');
h(1,2).CData(1,:) = cmap(ceil(256*0.6),:);%rgb('Green');
h(1,3).CData(1,:) = cmap(ceil(256*0.9),:)%rgb('Purple');
h(1,1).CData(2,:) = rgb('Black');
h(1,2).CData(2,:) = cmap(ceil(256*0.6),:);%rgb('Green');
h(1,3).CData(2,:) = cmap(ceil(256*0.9),:)%rgb('Purple');
xticklabels([roi_names])
%yticks([0:2:10])
%yticklabels([-0.1:0.5])
xlabel('ROI')
ylabel('Spearman Rank Correlation')
title('Similarity between depictions fMRI')

hold on
% Find the number of groups and the number of bars in each group

ngroups = size(all_sims, 1);
nbars = size(all_sims, 2);
% Calculate the width for each bar group

groupwidth = min(0.8, nbars/(nbars + 1.5));

% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, all_sims(:,i), all_se(:,i), 'k', 'linestyle', 'none');
end
legend({'Photo-Drawing'; 'Photo-Sketch'; 'Drawing-Sketch'} ,'Location','northeast')

print(fullfile(figure_path, ['RDM_sims_ROI_crossnobis.jpg']), ...
              '-djpeg', '-r300')

%% get noise ceiling using rsa toolbox 

addpath(genpath('/data/pt_02348/objdraw/fmri/rsatoolbox-1'))

[EVC_photo_upperBound, EVC_photo_lowerBound]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_photo_RDM,'Spearman',1);
[LO_photo_upperBound, LO_photo_lowerBound]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_photo_RDM,'Spearman',1);

[EVC_drawing_upperBound, EVC_drawing_lowerBound]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_drawing_RDM,'Spearman',1);
[LO_drawing_upperBound, LO_drawing_lowerBound]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_drawing_RDM,'Spearman',1);
  
[EVC_sketch_upperBound, EVC_sketch_lowerBound]=rsa.stat.ceilingAvgRDMcorr(EVC_fmri_sketch_RDM,'Spearman',1);
[LO_sketch_upperBound, LO_sketch_lowerBound]=rsa.stat.ceilingAvgRDMcorr(LOC_fmri_sketch_RDM, 'Spearman',1); 


  