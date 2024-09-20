%% function create_opt_fit_betas(beta_dir, outdir) 
%
%  This function takes the configurations for a subject and then searches
%  the residuals of the GLM with the 20 HRF fits and takes for each voxel
%  the betas with the HRf with the best fit and then creates new beta maps
%  with the only the betas from the HRF with the best fit for every voxel 
%

function create_opt_fit_betas(beta_dir, outdir)

% first load SPM file 

load(fullfile(beta_dir,'SPM.mat')); 

% then load mean residual map 

res = spm_read_vols(spm_vol(fullfile(beta_dir,'RPV.nii'))); 

% get the minimum indices for the residuals 

[~, min_ind] = min(res,4); 


% now loop over every regressor 

for i=1:x 
    
    % iteratively load the beta maps with the best fit and store the
    % corresponding beta for a given voxel 
    
end 




end 