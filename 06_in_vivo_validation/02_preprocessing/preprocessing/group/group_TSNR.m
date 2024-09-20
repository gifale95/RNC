%% aggregate TSNR measure across subjects 

clear all 
clc

% plotting defaults 
set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3, 'defaultaxesfontname', 'Helvetica') 

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

tsnr = NaN(length(fmri_subs),11);

for i_sub = 1:length(cfg.subject_indices)

    sub_id = fmri_subs{i_sub};
    
    results_dir =  fullfile(fmri_path,sub_id,'alldata', 'other'); 
    load(fullfile(results_dir, 'tsnr_across_runs.mat'));
    
    tsnr(i_sub,1:length(tsnr_mean)) = tsnr_mean;
end 

%% plot the individual TSNR series and the median in bold 

figure 
labels = {}; 
for  i_sub = 1:length(cfg.subject_indices)

     plot(tsnr(i_sub,:))
     hold on
     labels(i_sub) ={sprintf('Sub%02i',i_sub)};
end
plot(nanmedian(tsnr), 'black', 'LineWidth', 6)
labels(i_sub+1) = {'Median over subjects'};
xlim([1 11])
ylim([30 75])
title('TSNR across runs')
legend(labels, 'NumColumns',3)

print(fullfile(out_dir, ['group_TSNR.jpeg']), ...
             '-djpeg', '-r300')