%% qucik group fMRI Analysis 

clear all 
clc
%get config 
cfg = config_subjects_visdecmak; 

% set figurepath 
out_dir = '/Users/johannessinger/scratch/dfg_projekt/WP1/derived/exp/group';
fmri_path = '/Users/johannessinger/scratch/dfg_projekt/WP1/derived/exp/';

% add stats functions 
addpath(genpath('/Users/johannessinger/scratch/dfg_projekt/WP1/analysis/stats'))

% get fmri subnames 

fmri_subs = dir(fullfile(fmri_path,'*sub*'));
fmri_subs = {fmri_subs.name}';

% specify excluded subjects
excluded_subjects = {}; 

dth_corr = [];

% load behavior 
load(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived/', 'behav','RT_all_subjects_5_35_categorization.mat'), 'RTs')

mean_RTs = nanmedian(RTs,1); 

%specify results name
res_name = 'manmade_natural_hrf_fitting';

for i_sub = 1:length(cfg.subject_indices)
    
    sub_id = fmri_subs{i_sub};
    %if any(ismember(excluded_subjects, cfg.subject_indices(i_sub))), continue, end 
    
    if ~isdir(fullfile(fmri_path,sub_id, 'results','decoding',res_name,'roi')); fprintf('Not %i\n',i_sub);  continue; end; 
    
    results_dir =  fullfile(fmri_path,sub_id, 'results','decoding',res_name,'roi');
    load(fullfile(results_dir, 'res_mean_decision_values.mat'));
    
    for i = 1:length(results.mean_decision_values.output)
    
    these_dec_vals = results.mean_decision_values.output{i}; %reshape(results.mean_decision_values.output{i},2,60);
    if length(these_dec_vals) > 60
        these_dec_vals = mean(reshape(these_dec_vals,length(these_dec_vals)/60,60))';
    end 
    dth_corr(i_sub,i) = corr(these_dec_vals,mean_RTs', 'Type','Spearman'); 
end 

end 

fprintf('Mean distance to hyperplane correlation over all subjects EVC: %2f, LOC: %2f, PPA: %2f\n', mean(dth_corr(:,1)),mean(dth_corr(:,2)),mean(dth_corr(:,3))); 

%% plot 

roi_names = {'EVC'; 'LOC';'PPA'};

% set plot defaults 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3, 'defaultaxesfontname', 'Helvetica') 

%cmap = colormap('inferno');
%cmap = colormap('redblueTecplot');
close all

all_accs = mean(dth_corr)';
decoding_se = [std(dth_corr(:,1))/sqrt(length(dth_corr)),...
            std(dth_corr(:,3))/sqrt(length(dth_corr)),...
            std(dth_corr(:,3))/sqrt(length(dth_corr))]';

figure
h = bar(all_accs, 'grouped','FaceColor', 'flat');
h.CData= [1 0 0;0 0 1;0 1 0];
xticklabels([roi_names])
%yticks([0:-0.02:-0.12])
xlabel('ROI')
ylabel('Spearman R')
title('Distance To Hyperplane Correlation - fMRI')

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
    errorbar(x, all_accs(:,i), decoding_se(:,i), 'k', 'linestyle', 'none');
end
%legend({'Photos'; 'Drawings'; 'Sketches'} ,'Location','northeast')

print(fullfile(out_dir, ['manmade_natural_dth_corr_ROI.jpeg']), ...
             '-djpeg', '-r300')

%% compute statistics 

addpath(genpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/stats/'))

% set stats defaults 
nperm = 10000;
cluster_th = 0.001;
significance_th = 0.05;
tail = 'right';

sig_decoding_photo_EVC = permutation_1sample_alld (photo_group_decoding(:,1)-50, nperm, cluster_th, significance_th, tail);

sig_decoding_drawing_EVC = permutation_1sample_alld (drawing_group_decoding(:,1)-50, nperm, cluster_th, significance_th, tail);

sig_decoding_sketch_EVC = permutation_1sample_alld (sketch_group_decoding(:,1)-50, nperm, cluster_th, significance_th, tail);

sig_decoding_photo_LOC = permutation_1sample_alld (photo_group_decoding(:,2)-50, nperm, cluster_th, significance_th, tail);

sig_decoding_drawing_LOC = permutation_1sample_alld (drawing_group_decoding(:,2)-50, nperm, cluster_th, significance_th, tail);

sig_decoding_sketch_LOC = permutation_1sample_alld (sketch_group_decoding(:,2)-50, nperm, cluster_th, significance_th, tail);

% compute statistics on differences 

tail = 'both';

sig_photo_drawing_EVC = permutation_1sample_alld (photo_group_decoding(:,1)-drawing_group_decoding(:,1), nperm, cluster_th, significance_th, tail);

sig_photo_sketch_EVC = permutation_1sample_alld (photo_group_decoding(:,1)-sketch_group_decoding(:,1), nperm, cluster_th, significance_th, tail);

sig_drawing_sketch_EVC = permutation_1sample_alld (drawing_group_decoding(:,1)-sketch_group_decoding(:,1), nperm, cluster_th, significance_th, tail);


sig_photo_drawing_LOC = permutation_1sample_alld (photo_group_decoding(:,2)-drawing_group_decoding(:,2), nperm, cluster_th, significance_th, tail);

sig_photo_sketch_LOC = permutation_1sample_alld (photo_group_decoding(:,2)-sketch_group_decoding(:,2), nperm, cluster_th, significance_th, tail);

sig_drawing_sketch_LOC = permutation_1sample_alld (drawing_group_decoding(:,2)-sketch_group_decoding(:,2), nperm, cluster_th, significance_th, tail);

% control for multiple comparisons

[~,~,~,adj_p_EVC] = fdr_bh([sig_decoding_photo_EVC sig_decoding_drawing_EVC sig_decoding_sketch_EVC]);
[~,~,~,adj_p_LOC] = fdr_bh([sig_decoding_photo_LOC sig_decoding_drawing_LOC sig_decoding_sketch_LOC]);

[~,~,~,adj_p_diff_EVC] = fdr_bh([sig_photo_drawing_EVC sig_photo_sketch_EVC sig_drawing_sketch_EVC]);
[~,~,~,adj_p_diff_LOC] =  fdr_bh([sig_photo_drawing_LOC sig_photo_sketch_LOC sig_drawing_sketch_LOC]);
