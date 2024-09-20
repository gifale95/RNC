%% qucik group fMRI Analysis 

clear all 
clc

%setup paths
if ismac
fmri_path = '/Users/johannessinger/scratch/rcor_collab/derived/';

out_dir = '/Users/johannessinger/scratch/rcor_collab/derived/group';

elseif isunix 
    
   fmri_path = '/scratch/singej96/rcor_collab/derived/';

out_dir = '/scratch/singej96/rcor_collab/derived/group'; 
end 

% add stats functions 
addpath(genpath('/scratch/singej96/rcor_collab/analysis/stats'))

% get config 
addpath('/scratch/singej96/rcor_collab/analysis/')

cfg = config_subjects_rcor_alldata;
% get fmri subnames 

fmri_subs = dir(fullfile(fmri_path,'*sub*'));
fmri_subs = {fmri_subs.name}';

% specify excluded subjects
fmri_excluded_subs = {}; 

decoding_roi = [];

%specify results name
fname = 'res_accuracy_pairwise_minus_chance.mat'; %'s05wres_accuracy_minus_chance.nii' ;

% load searchlight results

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    decoding_roi_control_indiv = zeros(1,3);
    decoding_roi_challenge_indiv = zeros(1,3); 
    
    if ~any(ismember(fmri_excluded_subs,sub_id))
        
        % load fMRI RDMs
        fmri_fnames = cellstr(spm_select('fplistrec',fullfile(fmri_path,sub_id, 'results','decoding'),fname));

        if any(cellfun(@exist,fmri_fnames))
            fprintf('Loading fMRI %s\n',sub_id);
            
            for i=1:length(fmri_fnames)/4
                load(fmri_fnames{i});
                decoding_roi_challenge_indiv = decoding_roi_challenge_indiv+results.accuracy_pairwise_minus_chance.output';
            end 
            
            for i=length(fmri_fnames)/2+1:length(fmri_fnames)*0.75
                load(fmri_fnames{i});
                decoding_roi_control_indiv = decoding_roi_control_indiv+results.accuracy_pairwise_minus_chance.output';
            end 
            
        else
            fprintf('Results not complete for sub %s\n',sub_id);
            
        end
        
        decoding_roi_control(sub_no,:) = decoding_roi_control_indiv./nchoosek(10,2);
        decoding_roi_challenge(sub_no,:) = decoding_roi_challenge_indiv./nchoosek(10,2);

    end
end

fprintf('Mean decoding accuracies challenge over all subjects EVC: %2f, LOC: %2f, PPA: %2f\n', mean(decoding_roi_control(:,1)),mean(decoding_roi_control(:,3)),mean(decoding_roi_control(:,2))); 
fprintf('Mean decoding accuracies control over all subjects EVC: %2f, LOC: %2f, PPA: %2f\n', mean(decoding_roi_challenge(:,1)),mean(decoding_roi_challenge(:,3)),mean(decoding_roi_challenge(:,2))); 

%% plot 

roi_names = {'EVC'; 'PPA';'LOC'};

% set plot defaults 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3, 'defaultaxesfontname', 'Helvetica') 

%cmap = colormap('inferno');
%cmap = colormap('redblueTecplot');
close all

all_accs = [mean(decoding_roi_control); mean(decoding_roi_challenge)]';
decoding_se = [std(decoding_roi_control)/sqrt(length(decoding_roi_control));...
                std(decoding_roi_challenge)/sqrt(length(decoding_roi_challenge))]';


figure
h = bar(all_accs, 'grouped','FaceColor', 'flat');
h(1).CData= [0 0 1];
h(2).CData = [1 0 0];%0 1 0];
xticklabels([roi_names])
yticks([0:5:15])
yticklabels([50:5:65])
xlabel('ROI')
ylabel('Decoding Accuracy (%)')
title('Object Decoding - fMRI')

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
legend({'Control','Challenge'},'Location', 'northwest')

print(fullfile(out_dir, ['object_decoding_ROI.jpeg']), ...
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
