%% qucik group fMRI Analysis 

clear all 
clc

addpath(genpath('/data/pt_02348/objdraw/fmri/martin_spm'))
addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'))
addpath(genpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/meg/utils'))

% set figurepath 
figure_path = '/data/pt_02348/objdraw/group_level/fmri/';

% get config for experiment 

cfg = config_subjects_objdraw();

excluded_subjects = [7,9,10,13,22,23,29]; %od23because no mprage sequence there, 9, 10, 13 and 29 because bad data quality, 7,22 because missed a lot of catch trials 
photo_group_decoding= [];
drawing_group_decoding = [];
sketch_group_decoding = [];

res_name = 'final';

for i_sub = 1:length(cfg.subject_indices)

    if any(ismember(excluded_subjects, cfg.subject_indices(i_sub))), continue, end 
    
    if ~isdir(fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding',res_name)); fprintf('Not %i\n',i_sub);  continue; end; 
    
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

res_name = 'hrf_fitting';
photo_group_decoding_hrf_fitting = [];
drawing_group_decoding_hrf_fitting = [];
sketch_group_decoding_hrf_fitting = [];

for i_sub = 1:length(cfg.subject_indices)

    if any(ismember(excluded_subjects, cfg.subject_indices(i_sub))), continue, end 
    
    if ~isdir(fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding',res_name)); fprintf('Not %i\n',i_sub);  continue; end; 
    
    photo_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding',res_name,'roi','Photo');
    load(fullfile(photo_results_dir, 'res_accuracy_pairwise.mat'));
    photo_group_decoding_hrf_fitting = [results.accuracy_pairwise.output'; photo_group_decoding_hrf_fitting];
    drawing_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding',res_name,'roi','Drawing');
    load(fullfile(drawing_results_dir, 'res_accuracy_pairwise.mat'));
    drawing_group_decoding_hrf_fitting = [results.accuracy_pairwise.output'; drawing_group_decoding_hrf_fitting]; 
    sketch_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding',res_name,'roi','Sketch');
    load(fullfile(sketch_results_dir, 'res_accuracy_pairwise.mat'));
    sketch_group_decoding_hrf_fitting = [results.accuracy_pairwise.output'; sketch_group_decoding_hrf_fitting]; 

end 

fprintf('Mean photo decoding accuracies without HRF-fitting EVC: %2f, LOC: %2f\n', mean(photo_group_decoding(:,1)),mean(photo_group_decoding(:,2))); 
fprintf('Mean photo decoding accuracies with HRF-fitting EVC: %2f, LOC: %2f\n', mean(photo_group_decoding_hrf_fitting(:,1)),mean(photo_group_decoding_hrf_fitting(:,2)));
fprintf('Mean drawing decoding accuracies without HRF-fitting EVC: %2f, LOC: %2f\n', mean(drawing_group_decoding_hrf_fitting(:,1)),mean(drawing_group_decoding_hrf_fitting(:,2))); 
fprintf('Mean drawing decoding accuracies with HRF-fitting EVC: %2f, LOC: %2f\n', mean(drawing_group_decoding(:,1)),mean(drawing_group_decoding(:,2))); 
fprintf('Mean sketch decoding accuracies without HRF-fitting EVC: %2f, LOC: %2f\n', mean(sketch_group_decoding(:,1)),mean(sketch_group_decoding(:,2))); 
fprintf('Mean sketch decoding accuracies with HRF-fitting EVC: %2f, LOC: %2f\n', mean(sketch_group_decoding_hrf_fitting(:,1)),mean(sketch_group_decoding_hrf_fitting(:,2))); 

%% plot 

roi_names = {'EVC'; 'LOC+pFS'};

% set plot defaults 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3, 'defaultaxesfontname', 'Helvetica') 

%cmap = colormap('inferno');
cmap = colormap('redblueTecplot');
close all

all_accs = cat(2, mean(photo_group_decoding)', mean(photo_group_decoding_hrf_fitting)');
photo_se = [std(photo_group_decoding(:,1))/sqrt(length(photo_group_decoding)),...
            std(photo_group_decoding(:,2))/sqrt(length(photo_group_decoding))];
photo_hrf_fitting_se = [std(photo_group_decoding_hrf_fitting(:,1))/sqrt(length(photo_group_decoding_hrf_fitting)),...
            std(photo_group_decoding_hrf_fitting(:,2))/sqrt(length(photo_group_decoding_hrf_fitting))];
        
all_se = cat(2, photo_se',photo_hrf_fitting_se');

figure
h = bar(all_accs-50, 'grouped','FaceColor', 'flat');
h(1).CData = rgb('Black');
h(2).CData = cmap(ceil(256),:);
%h(3).CData = cmap(ceil(200),:);
xticklabels([roi_names])
yticks([0:5:35])
yticklabels([50:5:85])
xlabel('ROI')
ylabel('Decoding Accuracy (%)')
title('Category Decoding','Comparison with/without HRF-fitting')

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
legend({'No HRF-Fitting'; 'HRF-Fitting'} ,'Location','northeast')

print(fullfile(figure_path, ['decoding_with_without_HRF_fitting.jpg']), ...
              '-djpeg', '-r300')

