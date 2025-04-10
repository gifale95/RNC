%% combine results over iterations (for splithalf analysis) 

function combine_obj_decoding_results(results_dir, n_obj) 

all_accs = [];

% first load the results for all iterations
for i_obj = 1:n_obj
    for j_obj = 1+i_obj:n_obj
    
    cfg.results.dir = fullfile(results_dir,[num2str(i_obj),'_vs_',num2str(j_obj)]);
    
    acc = spm_read_vols(spm_vol(fullfile(cfg.results.dir, 'res_accuracy_pairwise_minus_chance.nii'))); 
    
    all_accs = cat(4,all_accs,acc); 
end  
end 

%average and write results 
vol = spm_vol(fullfile(cfg.results.dir, 'res_accuracy_pairwise_minus_chance.nii'));
vol.fname = fullfile(results_dir, 'mean_accuracy_pairwise_minus_chance.nii'); 
spm_write_vol(vol, mean(all_accs,4)); 

% %clear all the directories with the single results 
% for perm = 1:n_perm
%     
%     cfg.results.dir = fullfile(results_dir,num2str(perm));
%     rmdir(cfg.results.dir,'s')
%     
% end 
