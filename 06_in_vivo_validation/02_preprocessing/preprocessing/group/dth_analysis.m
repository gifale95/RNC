%% distance to hyperplane analysis 
function dth_analysis(cfg,i_sub,res_path)

load(fullfile(res_path,'res_mean_decision_values.mat'));

load(fullfile('/scratch/singej96/dfg_projekt/WP1/derived', 'behav','RT_all_subjects_5_35_categorization.mat'), 'RTs')

mean_RTs = nanmedian(RTs,1); 

for i = 1:length(results.mean_decision_values.output)
    
    these_dec_vals = results.mean_decision_values.output{i}; %reshape(results.mean_decision_values.output{i},2,60);
    if length(these_dec_vals) > 60
        these_dec_vals = mean(reshape(these_dec_vals,length(these_dec_vals)/60,60))';
    end 
    dth_corr(i) = corr(these_dec_vals,mean_RTs', 'Type','Spearman'); 
end 

% fill resultsvol 4D and write 4D nifi
backgroundvalue = NaN;
% get canonical hdr from first preprocesed functional file
template_file = dir(fullfile(cfg.sub(i_sub).dir, 'alldata','run01','*.nii'));
template_file = fullfile(template_file(1).folder,template_file(1).name);
hdr= spm_vol(template_file); % choose canonical hdr from first classification image
hdr = rmfield(hdr,'pinfo');
%hdr = rmfield(hdr, 'dt');

resultsvol_hdr = hdr;
resultsvol_hdr.fname = fullfile(res_path,'dth_corr.nii');
resultsvol_hdr.descrip = sprintf('Distance to hyperplane correlation with mean subject RT map');
resultsvol = backgroundvalue * ones(resultsvol_hdr.dim(1:3)); % prepare results volume with background value (default: 0)
resultsvol(results.mask_index) = dth_corr;
spm_write_vol(resultsvol_hdr,resultsvol);
end 
