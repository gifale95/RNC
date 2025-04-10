%% combine results over iterations (for splithalf analysis) 

function combine_decoding_results_splithalf(results_dir, n_perm) 

all_accs = [];

% first load the results for all iterations
for perm = 1:n_perm
    
    cfg.results.dir = fullfile(results_dir,num2str(perm));
    
    load(fullfile(cfg.results.dir, 'res_mean_decision_values.mat'))
    
    all_res_dec_val(perm,:) = results.mean_decision_values.output(:); 
    
    acc = spm_read_vols(spm_vol(fullfile(cfg.results.dir, 'res_accuracy_minus_chance.nii'))); 
    
    all_accs = cat(4,all_accs,acc); 
    
end 

% loop through searchlights and average results for each searchlight 
for i = 1:size(all_res_dec_val,2) 
    
    results.mean_decision_values.output{i} = mean(horzcat(all_res_dec_val{:,i}),2); 
    
end 

% save results 
save(fullfile(results_dir, 'res_mean_decision_values.mat'), 'results');

vol = spm_vol(fullfile(cfg.results.dir, 'res_accuracy_minus_chance.nii'));
vol.fname = fullfile(results_dir, 'res_accuracy_minus_chance.nii'); 
spm_write_vol(vol, mean(all_accs,4)); 

%clear all the directories with the single results 
for perm = 1:n_perm
    
    cfg.results.dir = fullfile(results_dir,num2str(perm));
    rmdir(cfg.results.dir,'s')
    
end 
