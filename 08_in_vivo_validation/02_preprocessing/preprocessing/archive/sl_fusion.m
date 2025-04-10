function results = decoding_similarity_crossnobis(condition_names,labels,beta_dir,out_dir,cfg)

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

% set everything to calculate (dis)similarity estimates
cfg.decoding.software = 'distance'; % the difference to 'similarity' is that this averages across data with the same label
cfg.decoding.method = 'classification'; % this is more a placeholder
cfg.decoding.train.classification.model_parameters = 'cveuclidean'; % cross-validated Euclidean after noise normalization

% This option below averages across (dis)similarity matrices of each
% cross-validation iteration and across all cells of the lower diagonal
% (i.e. all distance comparisons). If you want the entire matrix, consider
% using 'other_average' which only averages across cross-validation
% iterations. Alternatively, you could use the output 'RSA_beta' which is
% more general purpose, but a little more complex.
cfg.results.output = 'searchlight_fusion';

% These parameters carry out the multivariate noise normalization using the
% residuals
cfg.scale.method = 'cov'; % we scale by noise covariance
cfg.scale.estimation = 'separate'; % we scale all data for each run separately while iterating across searchlight spheres
cfg.scale.shrinkage = 'lw2'; % Ledoit-Wolf shrinkage retaining variances

% The crossnobis distance is identical to the cross-validated Euclidean
% distance after prewhitening (multivariate noise normalization). It has
% been shown that a good estimate for the multivariate noise is provided
% by the residuals of the first-level model, in addition with Ledoit-Wolf
% regularization. Here we calculate those residuals. If you have them
% available already, you can load them into misc.residuals using only the
% voxels from cfg.files.mask
[misc.residuals,cfg.files.residuals.chunk] = residuals_from_spm(fullfile(beta_dir,'SPM.mat'),cfg.files.mask); % this only needs to be run once and can be saved and loaded 


% Set additional parameters manually if you want (see decoding.m or
% decoding_defaults.m). Below some example parameters that you might want 
% to use:

cfg.searchlight.unit = 'mm';
cfg.searchlight.radius = 12; % this will yield a searchlight radius of 12mm.
cfg.searchlight.spherical = 1;
cfg.plot_design = 0;
cfg.verbose = 2;% want all information to be printed on screen


% Decide whether you want to see the searchlight/ROI/... during decoding
cfg.plot_selected_voxels = 0; % 0: no plotting, 1: every step, 2: every second step, 100: every hundredth step...


%% Nothing needs to be changed below for standard dissimilarity estimates using all data

% The following function extracts all beta names and corresponding run
% numbers from the SPM.mat
regressor_names = design_from_spm(beta_dir);

% Extract all information for the cfg.files structure (labels will be [1 -1] )
cfg = decoding_describe_data(cfg,condition_names,labels,regressor_names,beta_dir);

% This creates a design in which cross-validation is done between the distance estimates
cfg.design = make_design_similarity_cv(cfg);

% Run decoding
results = decoding(cfg,[],misc);
end 