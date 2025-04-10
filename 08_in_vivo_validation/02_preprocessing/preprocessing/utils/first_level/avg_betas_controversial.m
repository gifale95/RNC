%% this function averages the beta images for later RSA 

function avg_betas_controversial(labelnames,beta_dir,out_dir,cfg)

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
n_betas = 2; %number of betas we have for each condition 
shuffle_vector = randperm(n_betas);

for this_cat = 1: length(labelnames) 
    
    these_files = cfg.files.name(find(ismember(cfg.files.labelname, labelnames{this_cat})));
    
    % shuffle the files 
    these_files = these_files(shuffle_vector); 
    
    beta_all = [];
    
    % load all betas in one matrix 
    for this_file = 1:length(these_files) 
        
    vol = spm_vol(these_files{this_file});
    beta = spm_read_vols(vol);
    
    beta_all = cat(4,beta_all, beta); 
    
    end 
    
    beta_avg = mean(beta_all,4); 
    
    vol.fname = fullfile(out_dir,[labelnames{this_cat},'_beta_avg_' num2str(1) '.nii']);
    if ~exist(out_dir), mkdir(out_dir), end; 
    spm_write_vol(vol,beta_avg);
end        
end 