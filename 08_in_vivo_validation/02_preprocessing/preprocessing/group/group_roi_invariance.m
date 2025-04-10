%% qucik group fMRI Analysis 

clear all 
clc

addpath(genpath('/data/pt_02348/objdraw/fmri/martin_spm'))
addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'))

% set figurepath 
figure_path = '/data/pt_02348/objdraw/group_level/fmri/';

% get config for experiment 

cfg = config_subjects_objdraw();

excluded_subjects = [7,9,10,13,22,23,29]; %od23because no mprage sequence there, 9, 10, 13 and 29 because bad data quality, 7,22 because missed a lot of catch trials 
photo_group_decoding= [];
drawing_group_decoding = [];
sketch_group_decoding = [];

res_name = 'cat_noisenorm';

for i_sub = 1:length(cfg.subject_indices)

    if any(ismember(excluded_subjects, cfg.subject_indices(i_sub))), continue, end 
    
    photo_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding',res_name,'roi','Photo');
    load(fullfile(photo_results_dir, 'res_accuracy_pairwise.mat'));
    photo_group_decoding = [results.accuracy_pairwise.output'; photo_group_decoding]; 
    drawing_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding',res_name,'roi','Drawing');
    load(fullfile(drawing_results_dir, 'res_accuracy_pairwise.mat'));
    drawing_group_decoding = [results.accuracy_pairwise.output'; drawing_group_decoding]; 
    sketch_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding',res_name,'roi','Sketch');
    load(fullfile(sketch_results_dir, 'res_accuracy_pairwise.mat'));
    sketch_group_decoding = [results.accuracy_pairwise.output'; sketch_group_decoding]; 
end 

fprintf('Mean photo decoding accuracies over all subjects EVC: %2f, LOC: %2f\n', mean(photo_group_decoding(:,1)),mean(photo_group_decoding(:,2))); 
fprintf('Mean drawing decoding accuracies over all subjects EVC: %2f, LOC: %2f\n', mean(drawing_group_decoding(:,1)),mean(drawing_group_decoding(:,2))); 
fprintf('Mean sketch decoding accuracies over all subjects EVC: %2f, LOC: %2f\n', mean(sketch_group_decoding(:,1)),mean(sketch_group_decoding(:,2))); 

%run one sample t-test for all conditions and rois 
[photo_evc_h,photo_evc_p] = ttest(photo_group_decoding(:,1),50);
[photo_loc_h, photo_loc_p] = ttest(photo_group_decoding(:,2),50);

[drawing_evc_h,drawing_evc_p] = ttest(drawing_group_decoding(:,1),50);
[drawing_loc_h, drawing_loc_p] = ttest(drawing_group_decoding(:,2),50);

[sketch_evc_h,sketch_evc_p] = ttest(sketch_group_decoding(:,1),50);
[sketch_loc_h, sketch_loc_p] = ttest(sketch_group_decoding(:,2),50);

%% load cross decoding 

photo_drawing_group_decoding= [];
drawing_sketch_group_decoding = [];
photo_sketch_group_decoding = [];

res_name = 'cat_noisenorm_cvfold';

for i_sub = 1:length(cfg.subject_indices)

    if any(ismember(excluded_subjects, cfg.subject_indices(i_sub))), continue, end 
    
    photo_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','crossdecoding',res_name,'roi','Photo_Drawing');
    load(fullfile(photo_results_dir, 'res_accuracy_pairwise.mat'));
    photo_drawing_group_decoding = [results.accuracy_pairwise.output'; photo_drawing_group_decoding]; 
    drawing_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','crossdecoding',res_name,'roi','Drawing_Sketch');
    load(fullfile(drawing_results_dir, 'res_accuracy_pairwise.mat'));
    drawing_sketch_group_decoding = [results.accuracy_pairwise.output'; drawing_sketch_group_decoding]; 
    sketch_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','crossdecoding',res_name,'roi','Photo_Sketch');
    load(fullfile(sketch_results_dir, 'res_accuracy_pairwise.mat'));
    photo_sketch_group_decoding = [results.accuracy_pairwise.output'; photo_sketch_group_decoding]; 
end 

fprintf('Mean photo-drawing decoding accuracies over all subjects EVC: %2f, LOC: %2f\n', mean(photo_drawing_group_decoding(:,1)),mean(photo_drawing_group_decoding(:,2))); 
fprintf('Mean drawing-sketch decoding accuracies over all subjects EVC: %2f, LOC: %2f\n', mean(drawing_sketch_group_decoding(:,1)),mean(drawing_sketch_group_decoding(:,2))); 
fprintf('Mean photo-sketch decoding accuracies over all subjects EVC: %2f, LOC: %2f\n', mean(photo_sketch_group_decoding(:,1)),mean(photo_sketch_group_decoding(:,2))); 

%run one sample t-test for all conditions and rois 
[photo_evc_h,photo_evc_p] = ttest(photo_drawing_group_decoding(:,1),50);
[photo_loc_h, photo_loc_p] = ttest(photo_drawing_group_decoding(:,2),50);

[drawing_evc_h,drawing_evc_p] = ttest(drawing_sketch_group_decoding(:,1),50);
[drawing_loc_h, drawing_loc_p] = ttest(drawing_sketch_group_decoding(:,2),50);

[sketch_evc_h,sketch_evc_p] = ttest(photo_sketch_group_decoding(:,1),50);
[sketch_loc_h, sketch_loc_p] = ttest(photo_sketch_group_decoding(:,2),50);

%% compute differences

drawing_photo_group_decoding = drawing_group_decoding - photo_drawing_group_decoding;
sketch_photo_group_decoding = sketch_group_decoding - photo_sketch_group_decoding; 
sketch_drawing_group_decoding = drawing_group_decoding - drawing_sketch_group_decoding; 

%% plot 

% set plot defaults 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3) 

roi_names = {'EVC'; 'LOC'};

cmap = colormap('inferno');
close all

all_accs = cat(2, mean(drawing_photo_group_decoding)', mean(sketch_photo_group_decoding)', mean(sketch_drawing_group_decoding)');
drawing_photo_se = [std(drawing_photo_group_decoding(:,1))/sqrt(length(drawing_photo_group_decoding)),...
            std(drawing_photo_group_decoding(:,2))/sqrt(length(drawing_photo_group_decoding))];
sketch_photo_se = [std(sketch_photo_group_decoding(:,1))/sqrt(length(sketch_photo_group_decoding)),...
            std(sketch_photo_group_decoding(:,2))/sqrt(length(sketch_photo_group_decoding))];
sketch_drawning_se = [std(sketch_drawing_group_decoding(:,1))/sqrt(length(sketch_drawing_group_decoding)),...
            std(sketch_drawing_group_decoding(:,2))/sqrt(length(sketch_drawing_group_decoding))];
        
all_se = cat(2, drawing_photo_se', sketch_photo_se', sketch_drawning_se');

figure
h = bar(all_accs, 'grouped','FaceColor', 'flat');
h(1).CData = rgb('Black');
h(2).CData = cmap(ceil(256*0.3),:);
h(3).CData = cmap(ceil(256*0.6),:);
xticklabels([roi_names])
yticks([0:5:20])
yticklabels([0:5:20])
xlabel('ROI')
ylabel('Acc Difference(%)')
title('Invariance Analysis - fMRI')

hold on
% Find the number of groups and the number of bars in each group

ngroups = size(all_accs, 1);
nbars = size(all_accs, 2);
% Calculate the width for each bar group

groupwidth = min(0.8, nbars/(nbars + 1.5));

% Set the position of each error bar in the centre of the main bar
% Based on barweb.m by Bolu Ajiboye from MATLAB File Exchange
for i = 1:nbars
    % Calculate center of each bar
    x = (1:ngroups) - groupwidth/2 + (2*i-1) * groupwidth / (2*nbars);
    errorbar(x, all_accs(:,i), all_se(:,i), 'k', 'linestyle', 'none');
end
legend({'Drawing-Photo'; 'Sketch-Photo'; 'Drawing-Sketch'} ,'Location','northeast')

print(fullfile(figure_path, ['invariance_ROI_twoways.jpg']), ...
              '-djpeg', '-r300')

%% compute statistics 

addpath(genpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/stats/'))

% set stats defaults 
nperm = 10000;
cluster_th = 0.001;
significance_th = 0.05;
tail = 'right';

sig_decoding_drawing_photo_EVC = permutation_1sample_alld (drawing_photo_group_decoding(:,1), nperm, cluster_th, significance_th, tail);

sig_decoding_sketch_photo_EVC = permutation_1sample_alld (sketch_photo_group_decoding(:,1), nperm, cluster_th, significance_th, tail);

sig_decoding_sketch_drawing_EVC = permutation_1sample_alld (sketch_drawing_group_decoding(:,1), nperm, cluster_th, significance_th, tail);

sig_decoding_drawing_photo_LOC = permutation_1sample_alld (drawing_photo_group_decoding(:,2), nperm, cluster_th, significance_th, tail);

sig_decoding_sketch_photo_LOC = permutation_1sample_alld (sketch_photo_group_decoding(:,2), nperm, cluster_th, significance_th, tail);

sig_decoding_sketch_drawing_LOC = permutation_1sample_alld (sketch_drawing_group_decoding(:,2), nperm, cluster_th, significance_th, tail);

% compute statistics on differences 

tail = 'both';

sig_photo_drawing_photo_sketch_EVC = permutation_1sample_alld (sketch_photo_group_decoding(:,1)-drawing_photo_group_decoding(:,1), nperm, cluster_th, significance_th, tail);

sig_photo_drawing_sketch_drawing_EVC = permutation_1sample_alld (sketch_drawing_group_decoding(:,1)-drawing_photo_group_decoding(:,1), nperm, cluster_th, significance_th, tail);

sig_photo_sketch_drawing_sketch_EVC = permutation_1sample_alld (sketch_drawing_group_decoding(:,1)-sketch_photo_group_decoding(:,1), nperm, cluster_th, significance_th, tail);

sig_photo_drawing_photo_sketch_LOC = permutation_1sample_alld (sketch_photo_group_decoding(:,2)-drawing_photo_group_decoding(:,2), nperm, cluster_th, significance_th, tail);

sig_photo_drawing_sketch_drawing_LOC = permutation_1sample_alld (sketch_drawing_group_decoding(:,2)-drawing_photo_group_decoding(:,2), nperm, cluster_th, significance_th, tail);

sig_photo_sketch_drawing_sketch_LOC = permutation_1sample_alld (sketch_drawing_group_decoding(:,2)-sketch_photo_group_decoding(:,2), nperm, cluster_th, significance_th, tail);

% control for multiple comparisons

[~,~,~,adj_p_EVC] = fdr_bh([sig_decoding_drawing_photo_EVC sig_decoding_sketch_photo_EVC sig_decoding_sketch_drawing_EVC]);
[~,~,~,adj_p_LOC] = fdr_bh([sig_decoding_drawing_photo_LOC sig_decoding_sketch_photo_LOC sig_decoding_sketch_drawing_LOC]);

[~,~,~,adj_p_diff_EVC] = fdr_bh([sig_photo_drawing_photo_sketch_EVC sig_photo_drawing_sketch_drawing_EVC sig_photo_sketch_drawing_sketch_EVC]);
[~,~,~,adj_p_diff_LOC] =  fdr_bh([sig_photo_drawing_photo_sketch_LOC sig_photo_drawing_sketch_drawing_LOC sig_photo_sketch_drawing_sketch_LOC]);
