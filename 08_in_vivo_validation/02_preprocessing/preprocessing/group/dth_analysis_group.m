%% distance to hyperplane analysis 


for i_sub = 1:6
    
res_path = fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived',['sub0' num2str(i_sub)],'results','decoding','manmade_natural_hrf_fitting_normalized','searchlight');

load(fullfile(res_path,'res_mean_decision_values.mat'));

res_mask_index(i_sub) = {results.mask_index}; 

dth_vals(i_sub) = {results.mean_decision_values.output}; 

end 

load(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived', 'behav','RT_all_subjects_5_35_categorization.mat'), 'RTs')

%% normalize the RTs and average them 
% 
norm_RTs = zscore(RTs, 1,2); 

mean_RTs = nanmedian(RTs,1); 

%% format the decision values and compute dth corr

common_mask_idx = intersect(res_mask_index{1:2});

common_mask_idx = intersect(common_mask_idx, res_mask_index{3});

% fill resultsvol 4D and write 4D nifi
backgroundvalue = NaN;
% get canonical hdr from first preprocesed functional file
template_file = dir(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived/sub01/results/GLM/hrf_fitting/normalized','*.nii'));
template_file = fullfile(template_file(1).folder,template_file(1).name);
hdr= spm_vol(template_file); % choose canonical hdr from first classification image
hdr = rmfield(hdr,'pinfo');
resultsvol_hdr = hdr;
resultsvol_hdr.fname = fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived/group','dth_corr.nii');
resultsvol_hdr.descrip = sprintf('Distance to hyperplane correlation with mean subject RT map over subs 01 to 03');
resultsvol = backgroundvalue * ones(resultsvol_hdr.dim(1:3)); % prepare results volume with background value (default: 0)

for mask_idx = 1:length(common_mask_idx)
    
    these_dec_vals = (dth_vals{1}{find(res_mask_index{1}==common_mask_idx(mask_idx))}+...
                      dth_vals{2}{find(res_mask_index{2}==common_mask_idx(mask_idx))}+...
                      dth_vals{3}{find(res_mask_index{3}==common_mask_idx(mask_idx))})/3; %reshape(results.mean_decision_values.output{i},2,60);
                  
    resultsvol(common_mask_idx(mask_idx)) = corr(these_dec_vals,mean_RTs', 'Type','Spearman'); 
end 

spm_write_vol(resultsvol_hdr,resultsvol);