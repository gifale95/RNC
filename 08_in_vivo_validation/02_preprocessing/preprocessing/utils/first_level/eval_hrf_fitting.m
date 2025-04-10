function eval_hrf_fitting(res_dir) 

all_betas = []; 
all_residuals = [];
all_mean_residuals = []; 

%load SPM struct 

load(fullfile(res_dir,num2str(1), 'SPM.mat'));

    fprintf('Loading betas, residuals and mean residuals for all HRF models\n')

for hrf_idx = 1:20
    
    fprintf('HRF %i\n', hrf_idx)
    
    beta_names = dir(fullfile(res_dir,num2str(hrf_idx), '*beta*')); 
    
    beta_names = {beta_names.name}';
    
    for vol = 1:length(beta_names) 
        
    betas(:,:,:,vol) = spm_read_vols(spm_vol(fullfile(res_dir,num2str(hrf_idx),beta_names{vol}))); 
    
    if ~mod(vol,100); fprintf('.'); end
    
    end 
    
    fprintf('\n')
    
    res_names = dir(fullfile(res_dir,num2str(hrf_idx), '*Res_*')); 
    
    res_names = {res_names.name}';
    
    for vol = 1:length(res_names) 
        
    residuals(:,:,:,vol) = spm_read_vols(spm_vol(fullfile(res_dir,num2str(hrf_idx),res_names{vol}))); 
    
    if ~mod(vol,100); fprintf('.'); end

    end 
    
    fprintf('\n')
    
    mean_res_names = dir(fullfile(res_dir,num2str(hrf_idx), '*ResMS*')); 
    
    mean_res_names = {mean_res_names.name}';
    
    mean_residuals = spm_read_vols(spm_vol(fullfile(res_dir,num2str(hrf_idx),mean_res_names{:}))); 
    
    all_betas = cat(5,all_betas, betas);
    all_residuals = cat(5,all_residuals,residuals);
    all_mean_residuals = cat(4,all_mean_residuals,mean_residuals); 
    
end 


[~,HRFindex] = min(all_mean_residuals,[],4);

% take the corresponding values out of the residuals and the betas 

for vox = 1:length(mean_residuals(:))
    
    [x,y,z] = ind2sub(size(mean_residuals),vox); 
    
    sel_betas(x,y,z,:) = all_betas(x,y,z,:,HRFindex(x,y,z)); 
    sel_residuals(x,y,z,:) = all_residuals(x,y,z,:,HRFindex(x,y,z)); 
end 

clear all_betas all_residuals all_mean_residuals
% write selected betas and residuals to a new foler 

mkdir(res_dir,'fitted');

beta_hdr = spm_vol(fullfile(res_dir,num2str(1), beta_names{1})); 

disp('Writing fitted betas')

for vol = 1:size(beta_names) 
    
    beta_hdr.fname = fullfile(res_dir,'fitted',beta_names{vol});
    
    spm_write_vol(beta_hdr, sel_betas(:,:,:,vol)); 
   
    
end 

res_hdr = spm_vol(fullfile(res_dir,num2str(1),res_names{1})); 

disp('Writing fitted residuals')

for vol = 1:size(res_names) 
    
    
    res_hdr.fname = fullfile(res_dir,'fitted',res_names{vol});
    
    spm_write_vol(res_hdr, sel_residuals(:,:,:,vol));
    
end 

% save SPM mat to new folder 

save(fullfile(res_dir,'fitted','SPM.mat'), 'SPM'); 

% create a beta mask and save it to new folder

mask_hdr = spm_vol(fullfile(res_dir,num2str(1),'mask.nii')); 
mask = ~isnan(sel_betas(:,:,:,1));
mask_hdr.fname = fullfile(res_dir,'fitted','mask.nii');
spm_write_vol(mask_hdr,mask); 

% clean up the directories for disk space 

disp('Cleaning up')

for hrf_idx = 1:20
    
    try rmdir(fullfile(res_dir,num2str(hrf_idx)),'s'); 
    catch 
    end
    
end 

disp('Done.')

end 

