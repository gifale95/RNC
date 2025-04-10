function eval_hrf_fitting_optimized(res_dir,out_name) 

all_mean_residuals = []; 

%load SPM struct 

load(fullfile(res_dir,num2str(1), 'SPM.mat'));

fprintf('Loading mean residuals for all HRF models\n')
    
for hrf_idx = 1:20
      
    mean_res_names = dir(fullfile(res_dir,num2str(hrf_idx), '*ResMS*')); 
    
    mean_res_names = {mean_res_names.name}';
    
    mean_residuals = spm_read_vols(spm_vol(fullfile(res_dir,num2str(hrf_idx),mean_res_names{:}))); 
    
    all_mean_residuals = cat(4,all_mean_residuals,mean_residuals); 
    
end 

[~,HRFindex] = min(all_mean_residuals,[],4);
clear all_mean_residuals

% load contrasts
fprintf('Loading contrast for all HRF models\n')

con_names = dir(fullfile(res_dir,num2str(1), '*con*'));

con_names = {con_names.name}';

sz = size(spm_read_vols(spm_vol(fullfile(res_dir,num2str(1),con_names{1}))));

all_contrasts = NaN(sz(1),sz(2),sz(3),length(con_names),20);

for hrf_idx = 1:20
    
    parfor i = 1:length(con_names)
        
    con = spm_read_vols(spm_vol(fullfile(res_dir,num2str(hrf_idx),con_names{i})));
    
    all_contrasts(:,:,:,i,hrf_idx) = con; 
    end 
    
end

%load t-maps
fprintf('Loading t-maps for all HRF models\n')

t_names = dir(fullfile(res_dir,num2str(1), '*spmT*'));

t_names = {t_names.name}';

sz = size(spm_read_vols(spm_vol(fullfile(res_dir,num2str(1),t_names{1}))));

all_tmaps = NaN(sz(1),sz(2),sz(3),length(t_names),20);


% initialize all betas 

beta_names = dir(fullfile(res_dir,num2str(1), '*beta*'));

beta_names = {beta_names.name}';

sz = size(spm_read_vols(spm_vol(fullfile(res_dir,num2str(hrf_idx),beta_names{1})))); 

all_betas = NaN(sz(1),sz(2),sz(3),length(beta_names),20);

fprintf('Loading betas for all HRF models\n')

for hrf_idx = 1:20
    
    fprintf('HRF %i\n', hrf_idx)
    
    beta_names = dir(fullfile(res_dir,num2str(hrf_idx), '*beta*')); 
    
    beta_names = {beta_names.name}';
            
    parfor vol = 1:length(beta_names) 
        
    all_betas(:,:,:,vol,hrf_idx) = spm_read_vols(spm_vol(fullfile(res_dir,num2str(hrf_idx),beta_names{vol}))); 
    
    if ~mod(vol,100); fprintf('.'); end
    
    end 
end 

% take the corresponding values out of the residuals and the betas 

for vox = 1:length(mean_residuals(:))
    
    [x,y,z] = ind2sub(size(mean_residuals),vox); 
    
    sel_betas(x,y,z,:) = all_betas(x,y,z,:,HRFindex(x,y,z));
    sel_contrasts(x,y,z,:) = all_contrasts(x,y,z,:,HRFindex(x,y,z));
    sel_tmaps(x,y,z,:) = all_tmaps(x,y,z,:,HRFindex(x,y,z));
end 

clear all_betas

%initialize residuals 

res_names = dir(fullfile(res_dir,num2str(1), '*Res_*'));

res_names = {res_names.name}';

sz = size(spm_read_vols(spm_vol(fullfile(res_dir,num2str(1),res_names{1})))); 

sel_residuals = NaN(sz(1),sz(2),sz(3),length(res_names));

fprintf('Loading residuals for all HRF models\n')

for hrf_idx = 1:20
   
    fprintf('HRF %i\n', hrf_idx)
    
    res_names = dir(fullfile(res_dir,num2str(hrf_idx), '*Res_*')); 
    
    res_names = {res_names.name}';
    
    these_residuals = NaN(sz(1),sz(2),sz(3),length(res_names));
    
    parfor vol = 1:length(res_names) 
        
    these_residuals(:,:,:,vol) = spm_read_vols(spm_vol(fullfile(res_dir,num2str(hrf_idx),res_names{vol}))); 
    
    if ~mod(vol,100); fprintf('.'); end
    % take the corresponding values out of the residuals and the betas 
    end 
    
    for vox = 1:(sz(1)*sz(2)*sz(3))

        [x,y,z] = ind2sub([sz(1),sz(2),sz(3)],vox); 
        
        if HRFindex(x,y,z) == hrf_idx

        sel_residuals(x,y,z,:) = these_residuals(x,y,z,:); 
        end 
    end 
   
end 

%clear all_residuals

% write selected betas and residuals to a new folder 

mkdir(res_dir,out_name);

beta_hdr = spm_vol(fullfile(res_dir,num2str(1), beta_names{1})); 

disp('Writing fitted betas')

for vol = 1:size(beta_names) 
    
    beta_hdr.fname = fullfile(res_dir,out_name,beta_names{vol});
    
    spm_write_vol(beta_hdr, sel_betas(:,:,:,vol)); 
   
    
end 


con_hdr = spm_vol(fullfile(res_dir,num2str(1),con_names{1}));

disp('Writing fitted contrasts')

for vol = 1:size(con_names)
    
    
    con_hdr.fname = fullfile(res_dir,out_name,con_names{vol});
    
    spm_write_vol(con_hdr, sel_contrasts(:,:,:,vol));
    
end

t_hdr = spm_vol(fullfile(res_dir,num2str(1),t_names{1}));

disp('Writing fitted tmaps')

for vol = 1:size(t_names)
    
    
    t_hdr.fname = fullfile(res_dir,out_name,t_names{vol});
    
    spm_write_vol(t_hdr, sel_tmaps(:,:,:,vol));
    
end

res_hdr = spm_vol(fullfile(res_dir,num2str(1),res_names{1})); 

disp('Writing fitted residuals')

for vol = 1:size(res_names) 
    
    
    res_hdr.fname = fullfile(res_dir,out_name,res_names{vol});
    
    spm_write_vol(res_hdr, sel_residuals(:,:,:,vol));
    
end 

% save SPM mat to new folder 

save(fullfile(res_dir,out_name,'SPM.mat'), 'SPM'); 

% create a beta mask and save it to new folder

mask_hdr = spm_vol(fullfile(res_dir,num2str(1),'mask.nii')); 
mask = ~isnan(sel_betas(:,:,:,1));
mask_hdr.fname = fullfile(res_dir,out_name,'mask.nii');
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

