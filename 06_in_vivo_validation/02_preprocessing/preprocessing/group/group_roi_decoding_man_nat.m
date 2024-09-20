%% qucik group fMRI Analysis 

clear all 
clc

addpath(genpath('/data/pt_02348/objdraw/fmri/martin_spm'))
addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'))

% get config for experiment 

cfg = config_subjects_objdraw();

excluded_subjects = [7,12,13,22,23,29,31]; %od23 because no mprage sequence there, 12 and 29 because bad data quality

photo_group_decoding= [];
drawing_group_decoding = [];
sketch_group_decoding = [];

for i_sub = 1:length(cfg.subject_indices)

    if any(ismember(excluded_subjects, cfg.subject_indices(i_sub))), continue, end 
    
    photo_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding','cat','roi','Photo');
    load(fullfile(photo_results_dir, 'res_accuracy_pairwise.mat'));
    photo_group_decoding = [results.accuracy_pairwise.output'; photo_group_decoding]; 
    drawing_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding','cat','roi','Drawing');
    load(fullfile(drawing_results_dir, 'res_accuracy_pairwise.mat'));
    drawing_group_decoding = [results.accuracy_pairwise.output'; drawing_group_decoding]; 
    sketch_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding','cat','roi','Sketch');
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

%% plot 

roi_names = {'EVC'; 'LO'};

all_accs = cat(2, mean(photo_group_decoding)', mean(drawing_group_decoding)', mean(sketch_group_decoding)');
photo_se = [std(photo_group_decoding(:,1))/sqrt(length(photo_group_decoding)),...
            std(photo_group_decoding(:,2))/sqrt(length(photo_group_decoding))];
drawing_se = [std(drawing_group_decoding(:,1))/sqrt(length(drawing_group_decoding)),...
            std(drawing_group_decoding(:,2))/sqrt(length(drawing_group_decoding))];
sketch_se = [std(sketch_group_decoding(:,1))/sqrt(length(sketch_group_decoding)),...
            std(sketch_group_decoding(:,2))/sqrt(length(sketch_group_decoding))];
        
all_se = cat(2, photo_se', drawing_se', sketch_se');

figure
bar(all_accs-50, 'grouped')
xticklabels([roi_names])
yticks([0:2:10])
yticklabels([50:2:60])
xlabel('ROI')
ylabel('Mean Classification Accuracy')
title('Mean Classification Accuracy across ROIs')

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
    errorbar(x, all_accs(:,i)-50, all_se(:,i), 'k', 'linestyle', 'none');
end
legend({'Photos'; 'Drawings'; 'Sketches'} ,'Location','northeast')
