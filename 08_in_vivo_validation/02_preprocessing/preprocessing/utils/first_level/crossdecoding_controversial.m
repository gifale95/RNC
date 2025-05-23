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

function results = crossdecoding_controversial(condition_names,beta_nr, beta_dir, controversial_beta_dir, out_dir,cfg)

if ~exist('decoding_defaults.m','file')
    if ispc
        addpath('\\samba\hebart\decoding_toolbox');
    else
        addpath('/home/hebart/decoding_toolbox');
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

if cfg.noisenorm == 1 
    
    % These parameters carry out the multivariate noise normalization using the
% residuals
cfg.scale.method = 'cov'; % we scale by noise covariance
cfg.scale.estimation = 'separate'; % we scale all data for each run separately while iterating across searchlight spheres
cfg.scale.shrinkage = 'lw2'; % Ledoit-Wolf shrinkage retaining variances

[misc.residuals,cfg.files.residuals.chunk] = residuals_from_spm(fullfile(beta_dir,'SPM.mat'),cfg.files.mask); % this only needs to be run once and can be saved and loaded 

end 

% cfg.scale.method  = 'z';
% cfg.scale.estimation = 'all'; % we scale all data for each run separately while iterating across searchlight spheres
% 

cfg.searchlight.unit = 'mm';
cfg.searchlight.radius = 12; % this will yield a searchlight radius of 12mm.
cfg.searchlight.spherical = 1;
cfg.plot_design = 0;
cfg.verbose = 2; % you want all information to be printed on screen
% % libsvm settings
% defaults.decoding.train.classification.model_parameters = '-s 0 -t 0 -c 1 -b 1 -q'; % linear classification
% defaults.decoding.train.classification_kernel.model_parameters = '-s 0 -t 4 -c 1 -b 1 -q'; % linear classification
% defaults.decoding.test.classification.model_parameters = '-q -b 1';
% defaults.decoding.test.classification_kernel.model_parameters = '-q -b 1'; % linear classification
% cfg.results.output = {'accuracy_minus_chance','AUC_minus_chance'};

% Decide whether you want to see the searchlight/ROI/... during decoding
% cfg.plot_selected_voxels = 500; % 0: no plotting, 1: every step, 2: every second step, 100: every hundredth step...

% Add additional output measures if you like
cfg.results.output = {'decision_values','predicted_labels'}; 

%% Nothing needs to be changed below for a standard leave-one-run out cross
%% validation analysis.

% Set the following field:
% Full path to file names (1xn cell array) (e.g.
% {'c:\exp\glm\model_button\im1.nii', 'c:\exp\glm\model_button\im2.nii', ... }
betas = dir(fullfile(beta_dir,['*',condition_names{1}(1:end-2),'*']));
betas = {betas.name}';
betas = natsortfiles(betas); 
betas = fullfile(beta_dir,betas);
controversial_betas = dir(fullfile(controversial_beta_dir,['*',condition_names{1}(1:end-2),'*']));
controversial_betas = {controversial_betas.name}';
controversial_betas = natsortfiles(controversial_betas); 
controversial_betas = fullfile(controversial_beta_dir,controversial_betas)
all_betas = cat(1, betas(:),controversial_betas(:));
cfg.files.name = all_betas; 

% Other examples:
% For a cross classification, it would look something like this:
% cfg = decoding_describe_data(cfg,{labelname1classA labelname1classB labelname2classA labelname2classB},[1 -1 1 -1],regressor_names,beta_loc,[1 1 2 2]);
%
% set the two way option - if both directions should be classified or only
% one 
cfg.files.twoway = 0; 

% cfg.files.chunk = [repmat([1:5],1 ,60) repmat([6],1,60)]'; % little hack to pretend that all betas come from the same chunks (creates the right design) 
% % This creates the leave-one-run-out cross validation design:
% cfg.files.label = [ones(1,30*5)*-1, ones(1,30*5)*1 ones(1,30)*-1 ones(1,30)*1]';
% cfg.files.xclass = [ones(1,60*5) ones(1,60)*2];

cfg.files.chunk = [repmat([1:beta_nr],1 ,60) repmat([beta_nr+1],1,60)]'; % little hack to pretend that all betas come from the same chunks (creates the right design) 
% This creates the leave-one-run-out cross validation design:
cfg.files.label = [ones(1,30*beta_nr)*1, ones(1,30*beta_nr)*2 ones(1,30)*1 ones(1,30)*2]';
cfg.files.xclass = [ones(1,60*beta_nr) ones(1,60)*2];

cfg.design = make_design_xclass(cfg); 

% Run decoding
if cfg.noisenorm == 1
    results = decoding(cfg,[],misc);
else
    results = decoding(cfg);
end
end 