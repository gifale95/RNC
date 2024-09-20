%% this function applies univariate noise normalization to the beta images for later decoding 
function apply_unn_betas(labelnames,beta_dir,out_dir,cfg,i_sub)

% check if avg dir already exists 
if exist(out_dir)
    delete([out_dir, '/*.nii'])
elseif ~exist(out_dir)
    mkdir(out_dir)
end 

% get subject number 
sn = str2double(regexp(beta_dir,'(?<=sub).[0-9]','match'));

% load residuals for univariate noise normalization
res_names = dir(fullfile(beta_dir, '*Res_*')); 
res_names = {res_names.name}';

residuals = residuals_without_spm(fullfile(beta_dir,res_names),fullfile(beta_dir,'mask.nii')); 

% since the labels are arbitrary, we will set them randomly to -1 and 1
labels = [1:length(labelnames)];

% The following function extracts all beta names and corresponding run
% numbers from the SPM.mat
regressor_names = design_from_spm(beta_dir);

% Extract all information for the cfg.files structure (labels will be [1 -1] )
cfg = decoding_describe_data(cfg,labelnames,labels,regressor_names,beta_dir);

% load mask for indexing betas 
mask = spm_read_vols(spm_vol(fullfile(beta_dir,'mask.nii')));

% specify which residual images belong to which run
chunk = repelem([1:cfg.sub(i_sub).n_runs], cfg.n_scans_experiment);

for this_cat = 1: length(labelnames) 
    
    these_files = cfg.files.name(find(ismember(cfg.files.labelname, labelnames{this_cat})));
    
    % load all betas in one matrix 
    for this_file = 1:length(these_files) 
        
    
    vol = spm_vol(these_files{this_file});
    beta = spm_read_vols(vol);
    % mask beta to fit size of residuals
    masked_beta = beta(logical(mask));
    % compute std deviation of residuals
    res_sd = std(residuals(chunk==this_file,:));
    % normalize beta voxelwise by the std of the residuals
    norm_beta = masked_beta'./res_sd; 
    % reshape the normalized betas to the size of the volume
    sz = size(beta);
    norm_beta_re = zeros(sz(1),sz(2),sz(3));
    norm_beta_re(logical(mask))= norm_beta; 
    % get the path for the new files 
    split_string = split(these_files{this_file},'/');
    fname = split_string{end};
    fname = fullfile(out_dir,fname);
    % change the header of the volume and write volume 
    vol.fname = fname;
    spm_write_vol(vol,norm_beta_re); 
    
    end 
end        
end 