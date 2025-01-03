% function decoding_sl(condition_names,labels,beta_dir,out_dir,cfg)
%
% Wrapper script for decoding using a searchlight.
%
% Input variables:
%   condition_names: Names of all regressors to be used for classification
%   labels: Labels that should be paired with condition_names (e.g. [-1 1])
%   beta_dir: name where results are which are used for classification
%   out_dir: name of folder where results are saved
%   cfg (optional): config variable used for the decoding

function results = decoding_sl(condition_names,labels,beta_dir,out_dir,cfg,cfg_sub,i_sub)

if ~exist('decoding_defaults.m','file')
    if ismac
        addpath('/Users/johannessinger/Documents/cloud_Berlin/Projekte/dfg/WP1/analysis_tools/tdt_3.999/decoding_toolbox');
    else
        addpath('/scratch/singej96/code_checking/reading/tdt_3.999/decoding_toolbox');
    end
end

if ~exist('cfg','var')
    cfg = decoding_defaults;
else
    cfg = decoding_defaults(cfg);
end

% Get subject number
try
    cfg.sn = str2double(regexp(beta_dir,'(?<=sub).[0-9]','match'));
catch
    warning('Could not extract subject number.')
end

% Set the analysis that should be performed (default is 'searchlight')
%cfg.analysis = 'searchlight';

% Set the output directory where data will be saved, e.g. 'c:\exp\results\buttonpress'
cfg.results.dir = out_dir;
cfg.results.overwrite = 1;

% Set the filepath where your SPM.mat and all related betas are, e.g. 'c:\exp\glm\model_button'
% done already

% Set the filename of your brain mask (or your ROI masks as cell matrix) 
% for searchlight or wholebrain e.g. 'c:\exp\glm\model_button\mask.img' OR 
% for ROI e.g. {'c:\exp\roi\roimaskleft.img', 'c:\exp\roi\roimaskright.img'}
try cfg.files.mask;
catch
    cfg.files.mask = fullfile(beta_dir,'mask.img');
    if ~exist(cfg.files.mask,'file')
        cfg.files.mask = fullfile(beta_dir,'mask.nii');
        if ~exist(cfg.files.mask,'file')
            error('Mask not found in %s',cfg.files.mask)
        end
    end
end

% Set additional parameters manually if you want (see decoding.m or
% decoding_defaults.m). Below some example parameters that you might want 
% to use:

% in case similarities should be calculated
if strcmpi(cfg.decoding.software,'similarity')
    cfg.decoding.method = 'classification';
end 


if  strcmpi(cfg.decoding.software, 'liblinear')
%use this for liblinear - might be faster for big datasets 
cfg.decoding.method = 'classification';
cfg.decoding.train.classification.model_parameters = '-s 1 -c 1 -q';
end 
 
if cfg.noisenorm == 1 
    
    % These parameters carry out the multivariate noise normalization using the
% residuals
cfg.scale.method = 'cov'; % we scale by noise covariance
cfg.scale.estimation = 'separate'; % we scale all data for each run separately while iterating across searchlight spheres
cfg.scale.shrinkage = 'lw2'; % Ledoit-Wolf shrinkage retaining variances

if ~cfg.hrf_fitting
    
    [misc.residuals,cfg.files.residuals.chunk] = residuals_from_spm(fullfile(beta_dir,'SPM.mat'),cfg.files.mask); % this only needs to be run once and can be saved and loaded 
    
elseif cfg.hrf_fitting 
        
    res_names = dir(fullfile(beta_dir, '*Res_*')); 
    
    res_names = {res_names.name}';
    
%     for vol = 1:length(res_names) 
%         
%     misc.residuals(:,:,:,vol) = spm_read_vols(spm_vol(fullfile(beta_dir,res_names{vol}))); 
%     
%     end 
%     
%     % take only the residuals in the mask regions 
%     
%     EVC_mask = [];

    misc.residuals = residuals_without_spm(fullfile(beta_dir,res_names),cfg.files.mask); 
    
    cfg.files.residuals.chunk = repelem([1:cfg_sub.sub(i_sub).n_betas], cfg_sub.n_scans_experiment); 
end 
    
end 

cfg.searchlight.unit = 'voxels';
cfg.searchlight.radius = 3; % this will yield a searchlight radius of 12mm or 10mm 
cfg.searchlight.spherical = 1;
cfg.plot_design = 0;
cfg.verbose = 2; % you want all information to be printed on screen
%cfg.decoding.train.classification.model_parameters = '-s 0 -t 0 -c 1 -b 0 -q'; 
cfg.results.output = {'accuracy_minus_chance'};

% Decide whether you want to see the searchlight/ROI/... during decoding
% cfg.plot_selected_voxels = 500; % 0: no plotting, 1: every step, 2: every second step, 100: every hundredth step...

% Add additional output measures if you like
% cfg.results.output = {'accuracy_minus_chance', 'AUC'}

%% Nothing needs to be changed below for a standard leave-one-run out cross
%% validation analysis.

% The following function extracts all beta names and corresponding run
% numbers from the SPM.mat
regressor_names = design_from_spm(beta_dir);

% Extract all information for the cfg.files structure (labels will be [1 -1] )
cfg = decoding_describe_data(cfg,condition_names,labels,regressor_names,beta_dir);

% This creates the leave-one-run-out cross validation design:
cfg.design = make_design_cv(cfg); 
cfg.design.unbalanced_data = 'ok';  

if ismac | cfg.parallel ==0
% Run decoding
if cfg.noisenorm 
    results = decoding(cfg,[],misc);
else
    results = decoding(cfg);
end 

elseif isunix | cfg.parallel ==1
    
    % get the number of searchlights for parallelization 
    mask = spm_read_vols(spm_vol(cfg.files.mask{1}));
    num_searchlights = size(mask(mask==1),1);
    % start the parallel pool
    pool =  parcluster('local');
    % get number of workers (cpus)
    num_workers = pool.NumWorkers;
    searchlights_per_worker = ceil(num_searchlights/num_workers); % Divide the task up into the number of workers
    fprintf('Starting searchlight decoding on %i parallel cores with %i searchlights for each worker', num_workers, searchlights_per_worker)
    if ~cfg.noisenorm
            misc = [];
    end 
    parfor crun = 1:num_workers
            results{crun} = decoding_parallel_wrapper(cfg,searchlights_per_worker,crun,misc);
    end
    all_results = results{1};
    for crun = 2:num_workers
    all_results.decoding_subindex = [all_results.decoding_subindex; results{crun}.decoding_subindex];
    all_results.accuracy_minus_chance.output(results{crun}.decoding_subindex) = results{crun}.accuracy_minus_chance.output(results{crun}.decoding_subindex); 
    %all_results.other_average.output(results{crun}.decoding_subindex) = results{crun}.other_average.output(results{crun}.decoding_subindex);
    end
    results = all_results;
    disp('Decoding on the whole brain complete, saving results')
    sub_config = config_subjects_visdecmak(); 
    combine_write_decoding_results(sub_config, i_sub,results, fullfile(cfg.results.dir,'res_accuracy_minus_chance.nii'))
    save(fullfile(cfg.results.dir,'res_accuracy_minus_chance.mat'),'results')
    %assert(sum(cellfun(@isempty,all_results.oth.output))==0,‘Results Output not completely filled despite completion of the parallel loop - please check’)
    delete(fullfile(cfg.results.dir,'parallel_loop*.mat'))
    delete(fullfile(cfg.results.dir,'parallel_loop*.nii'))

end 
end