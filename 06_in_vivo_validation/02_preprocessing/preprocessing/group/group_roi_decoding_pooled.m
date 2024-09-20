%% qucik group fMRI Analysis 

clear all 
clc

addpath(genpath('/data/pt_02348/objdraw/fmri/martin_spm'))
addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'))

% set figurepath 
figure_path = '/data/pt_02348/objdraw/group_level/fmri/';

% add some plotting utils 

addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/meg/utils');

% get config for experiment 

cfg = config_subjects_objdraw();

excluded_subjects = [7,12,13,22,23,29,31]; %od23 because no mprage sequence there, 12 and 29 because bad data quality

group_decoding= [];

for i_sub = 1:length(cfg.subject_indices)

    if any(ismember(excluded_subjects, cfg.subject_indices(i_sub))), continue, end 
    
    photo_results_dir = fullfile(cfg.sub(cfg.subject_indices(i_sub)).dir,'results','decoding','cat_denoise','roi','all');
    load(fullfile(photo_results_dir, 'res_accuracy_pairwise.mat'));
    group_decoding = [results.accuracy_pairwise.output'; group_decoding]; 

end 

fprintf('Mean decoding accuracies over all subjects EVC: %2f, LOC: %2f\n', mean(group_decoding(:,1)),mean(group_decoding(:,2))); 


%% plot 

roi_names = {'EVC'; 'LOC'};

all_accs = mean(group_decoding)';
all_se = [std(group_decoding(:,1))/sqrt(length(group_decoding)),...
            std(group_decoding(:,2))/sqrt(length(group_decoding))]';
     
figure
h = bar(all_accs-50,'grouped', 'FaceColor', 'flat');
h.CData(1,:) = rgb('Black');
h.CData(2,:) = rgb('Green');
xticklabels([roi_names])
yticks([0:2:30])
yticklabels([50:2:80])
xlabel('ROI')
ylabel('Accuracy')
title('Category Decoding across ROIs')

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
%legend({'Photos'; 'Drawings'; 'Sketches'} ,'Location','northeast')

%print(fullfile(figure_path, ['cat_crossdecoding_ROI.jpg']), ...
%              '-djpeg', '-r300')