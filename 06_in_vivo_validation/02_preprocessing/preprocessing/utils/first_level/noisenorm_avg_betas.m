%% this function averages the beta images for later RSA 
function noisenorm_avg_betas(labelnames, avg_size,beta_dir,out_dir,n_runs, mask,i_sub, cfg)

% check if avg dir already exists 
if exist(out_dir)
    delete([out_dir, '/*.nii'])
end 

% since the labels are arbitrary, we will set them randomly to -1 and 1
labels = [ones(1,30) ones(1,30)*2];

% The following function extracts all beta names and corresponding run
% numbers from the SPM.mat
regressor_names = design_from_spm(beta_dir);

% Extract all information for the cfg.files structure (labels will be [1 -1] )
cfg = decoding_describe_data(cfg,labelnames,labels,regressor_names,beta_dir);

% create the shuffle vector before so its the same for all conditions 
n_betas = n_runs; %number of betas we have for each condition 
shuffle_vector = randperm(n_betas);

% load the residuals for noise-normalizing the data 
res_names = dir(fullfile(beta_dir, '*Res_*'));

res_names = {res_names.name}';

residuals = residuals_without_spm(fullfile(beta_dir,res_names),mask);

residuals_chunk = repelem([1:cfg.sub(i_sub).n_betas], 311);

residuals = residuals(1:length(residuals_chunk),:); 

beta_all = [];

% prepare mask for masking betas 
mask = spm_read_vols(spm_vol(mask{:})); 
mask_index = find(mask==1); 

for chunk = 1:cfg.sub(i_sub).n_betas
    
    % select only residuals from corresponding run
    these_residuals = residuals(residuals_chunk == chunk,:);
    % get the sigma for the residuals
    fprintf('Computing sigma for chunk %i\n', chunk)
    [sigma,lambda] = covshrink_lw2(these_residuals);

    for this_cat = 1: length(labelnames)
        
        these_files = cfg.files.name(find(ismember(cfg.files.labelname, labelnames{this_cat})));
        
        % load all betas in one matrix
        hdr = spm_vol(these_files{chunk});
        vol = spm_read_vols(hdr);
        vol = vol(logical(mask));
        %vol = vol(1:10000); 
        % scale betas with the sigma for this run
        beta_all(this_cat,chunk,:) = vol'*sigma^(-1/2);
    end 
end 

disp('Averaging scaled betas')
% permute the run order 
beta_all = beta_all(:,randperm(size(beta_all,2)),:); 

%average 2 betas into one beta
ct = 1; 
for i=1:avg_size:length(these_files)
    
    beta_avg = squeeze(mean(beta_all(:,i:i+(avg_size-1),:),2));
    
    for this_cat = 1:length(labelnames)
        hdr.fname = fullfile(out_dir,[labelnames{this_cat},'_beta_avg_' num2str(ct) '.nii']);
        if ~exist(out_dir), mkdir(out_dir), end;
        this_beta_avg= zeros(size(mask,1),size(mask,2),size(mask,3)); 
        this_beta_avg(mask_index) = beta_avg(this_cat,:); 
        spm_write_vol(hdr,this_beta_avg);
    end 
    ct = ct+1;
end
end