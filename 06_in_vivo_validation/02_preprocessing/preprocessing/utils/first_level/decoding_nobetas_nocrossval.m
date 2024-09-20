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

function results = decoding_nobetas(condition_names,avg_size,beta_dir,out_dir,cfg,i_sub)

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
end 

% % scaling parameters if desired 
% cfg.scale.method = 'min0max1'; 
% cfg.scale.estimation = 'separate'; 

cfg.searchlight.unit = 'mm'; %'mm'
cfg.searchlight.radius = 12; % 12; this will yield a searchlight radius of 12mm or 10mm 
cfg.searchlight.spherical = 1;
cfg.plot_design = 0;
cfg.verbose = 2; % you want all information to be printed on screen

% cfg.decoding.train.classification.model_parameters = '-s 0 -t 0 -c 1 -b 0 -q'; 
cfg.results.output = {'accuracy_minus_chance','mean_decision_values'};

% Decide whether you want to see the searchlight/ROI/... during decoding
% cfg.plot_selected_voxels = 500; % 0: no plotting, 1: every step, 2: every second step, 100: every hundredth step...

% Add additional output measures if you like
% cfg.results.output = {'accuracy_minus_chance', 'AUC'}

%% Nothing needs to be changed below for a standard leave-one-run out cross
%% validation analysis.

% Set the following field:
% Full path to file names (1xn cell array) (e.g.
% {'c:\exp\glm\model_button\im1.nii', 'c:\exp\glm\model_button\im2.nii', ... }
betas = dir(fullfile(beta_dir,'avg',['*',condition_names{1}(1:end-2),'*']));
betas = {betas.name}';
betas = natsortfiles(betas); 
betas = fullfile(beta_dir,'avg',betas);

betas_test = dir(fullfile(beta_dir,'avg_test','*.nii'));
betas_test = {betas_test.name}';
betas_test = natsortfiles(betas_test); 
betas_test = fullfile(beta_dir,'avg_test',betas_test);


all_betas = cat(1, betas(:),betas_test(:));
cfg.files.name = all_betas;

% Other examples:
% For a cross classification, it would look something like this:
% cfg = decoding_describe_data(cfg,{labelname1classA labelname1classB labelname2classA labelname2classB},[1 -1 1 -1],regressor_names,beta_loc,[1 1 2 2]);
%
% set the two way option - if both directions should be classified or only
% one 
cfg.files.twoway = 0; 

cfg.files.chunk = [repmat([1:avg_size],1 ,60) repmat([avg_size+1],1,60)]'; % little hack to pretend that all betas come from the same chunks (creates the right design) 
% This creates the leave-one-run-out cross validation design:
cfg.files.label = [ones(1,30*avg_size), ones(1,30*avg_size)*-1 ones(1,30) ones(1,30)*-1]';
cfg.files.xclass = [ones(1,60*avg_size) ones(1,60)*2];

cfg.design = make_design_xclass(cfg); 

if ismac | cfg.parallel ==0
% Run decoding
    if cfg.noisenorm == 1
    results = decoding(cfg,[],misc);
    else
    results = decoding(cfg);
    end 

elseif isunix && cfg.parallel ==1
    
    % get the number of searchlights for parallelization 
    mask = spm_read_vols(spm_vol(cfg.files.mask{1}));
    num_searchlights = size(mask(mask==1),1);
    % start the parallel pool
    pool =  parcluster('local');
    % get number of workers (cpus)
    num_workers = pool.NumWorkers;
    searchlights_per_worker = ceil(num_searchlights/num_workers); % Divide the task up into the number of workers
    fprintf('Starting searchlight decoding on %i parallel cores with %i searchlights for each worker', num_workers, searchlights_per_worker)
    parfor crun = 1:num_workers
    results{crun} = decoding_parallel_wrapper(cfg,searchlights_per_worker,crun)
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