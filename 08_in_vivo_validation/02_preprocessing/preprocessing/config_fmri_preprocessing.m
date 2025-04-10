function cfg = config_fmri_preprocessing(cfg)

% Sets analysis specific variables

%% Set experimental name (= experiment folder) and file prefix

cfg.name = 'rnc';
cfg.prefix = [];
cfg.prefix_localizer = 'loc'; 
cfg.suffix = 'nii'; % img or nii (tested with img)

%% Set and add paths
data_dir = '../relational_neural_control/in_vivo_validation/in_vivo_fmri_dataset/raw/'; % Download raw fMRI from: https://openneuro.org/datasets/ds005503
spm_dir = '../spm12/'; % Download from: https://www.fil.ion.ucl.ac.uk/spm/software/download/
home_dir = '../relational_neural_control/in_vivo_validation/in_vivo_fmri_dataset/';
derived_dir = '../relational_neural_control/in_vivo_validation/in_vivo_fmri_dataset/derived/';
code_dir = '../06_in_vivo_validation/02_preprocessing/preprocessing/';

cfg.dirs.data_dir = data_dir;
cfg.dirs.spm_dir = spm_dir;
cfg.dirs.home_dir = home_dir;
cfg.dirs.derived_dir = derived_dir;
cfg.dirs.code_dir = code_dir;


%% Set scanner-specific and basic analysis variables

cfg.TE = 0.033;
cfg.TR = 1;
cfg.fieldmapTE = [4.92 7.38];
cfg.echo_readout_time = 0.0534613*1000; % 108 phase encoding steps * echo spacing / grappa acceleration --> you can also get that from the .json files 
cfg.n_slices = 39;
cfg.voxdims = [2.5 2.5 2.5];
cfg.dim = [82 82 39]; % voxels n the x dimension y dimension z dimension
cfg.sliceorder = 'multiband'; % can be ascending, descending, interleaved, interleaved descending or multiband
cfg.slicetiming = [
		0,
		0.53,
		0.0775,
		0.605,
		0.1525,
		0.68,
		0.2275,
		0.755,
		0.3025,
		0.83,
		0.3775,
		0.905,
		0.4525,
		0,
		0.53,
		0.0775,
		0.605,
		0.1525,
		0.68,
		0.2275,
		0.755,
		0.3025,
		0.83,
		0.3775,
		0.905,
		0.4525,
		0,
		0.53,
		0.0775,
		0.605,
		0.1525,
		0.68,
		0.2275,
		0.755,
		0.3025,
		0.83,
		0.3775,
		0.905,
		0.4525	]; 
cfg.FWHM = 5; % smoothing factor in mm 

%% Set subjects to use for analysis

excluded_subjects = []; % excluded because measurement was interrupted
  
excluded_runs = {}; % no runs excluded


%% Set import specific variables

cfg.n_dummy = 0; % number of dummy scans to remove in beginning of each run
cfg.n_scans_prf_experiment = 300; % prf experiment volumes
cfg.n_scans_uc_experiment= 436; % univariate RNC experiment volumes
cfg.n_scans_mc_experiment= 484; % multivariate RNC experiment volumes
cfg.n_scans_localizer = 0; 

%% Set Subject Variables

% entries are: scaled variance of image, scaled variance of slice, deviation of 0.2 mm, rotation of 0.01 deg
outlier_cutoff = [5 1000 0.5 4/120]; %30 2000 0.5 1/120]; % default value, change only in subjects if necessary

% Subject 1
cfg.sub(1).id = 'sub-01';
cfg.sub(1).nr = 1;
cfg.sub(1).age = [23];
cfg.sub(1).gender = 1; % 1 is female, 0 is male
cfg.sub(1).import.prf_experiment_runs = 3; % prf experiment runs
cfg.sub(1).import.uc_experiment_runs = 10; % univariate RNC experiment runs
cfg.sub(1).import.mc_experiment_runs = 12; % multivariate RNC experiment runs
cfg.sub(1).import.localizer = [];
cfg.sub(1).import.second_localizer = [];
cfg.sub(1).preproc.outlier_cutoff = outlier_cutoff; % first value: image difference, second value: slice difference in variance
cfg.sub(1).preproc.corrupted_volumes = []; % Numerical format: A.xy where A refers to the total volume number and xy to the slice number. Example: 768.03 means volume 768, slice 03
cfg.sub(1).preproc.corrupted_slices = [];
cfg.sub(1).preproc.eyesize_range = [80 1000 0.8];
cfg.sub(1).dicomtime = []; % we didn't record the first run

% Subject 2
cfg.sub(2).id = 'sub-02';
cfg.sub(2).nr = 2;
cfg.sub(2).age = [24];
cfg.sub(2).gender = 1; % 1 is female, 0 is male
cfg.sub(2).import.prf_experiment_runs = 3; % prf experiment runs
cfg.sub(2).import.uc_experiment_runs = 10; % univariate RNC experiment runs
cfg.sub(2).import.mc_experiment_runs = 12; % multivariate RNC experiment runs
cfg.sub(2).import.localizer = [];
cfg.sub(2).import.second_localizer = [];
cfg.sub(2).preproc.outlier_cutoff = outlier_cutoff; % first value: image difference, second value: slice difference in variance
cfg.sub(2).preproc.corrupted_volumes = []; % Numerical format: A.xy where A refers to the total volume number and xy to the slice number. Example: 768.03 means volume 768, slice 03
cfg.sub(2).preproc.corrupted_slices = [];
cfg.sub(2).preproc.eyesize_range = [80 1000 0.8];
cfg.sub(2).dicomtime = []; % we didn't record the first run

% Subject 3
cfg.sub(3).id = 'sub-03';
cfg.sub(3).nr = 3;
cfg.sub(3).age = [25];
cfg.sub(3).gender = 1; % 1 is female, 0 is male
cfg.sub(3).import.prf_experiment_runs = 3; % prf experiment runs
cfg.sub(3).import.uc_experiment_runs = 10; % univariate RNC experiment runs
cfg.sub(3).import.mc_experiment_runs = 12; % multivariate RNC experiment runs
cfg.sub(3).import.localizer = [];
cfg.sub(3).import.second_localizer = [];
cfg.sub(3).preproc.outlier_cutoff = outlier_cutoff; % first value: image difference, second value: slice difference in variance
cfg.sub(3).preproc.corrupted_volumes = []; % Numerical format: A.xy where A refers to the total volume number and xy to the slice number. Example: 768.03 means volume 768, slice 03
cfg.sub(3).preproc.corrupted_slices = [];
cfg.sub(3).preproc.eyesize_range = [80 1000 0.8];
cfg.sub(3).dicomtime = []; % we didn't record the first run

% Subject 4
cfg.sub(4).id = 'sub-04';
cfg.sub(4).nr = 4;
cfg.sub(4).age = [30];
cfg.sub(4).gender = 0; % 1 is female, 0 is male
cfg.sub(4).import.prf_experiment_runs = 3; % prf experiment runs
cfg.sub(4).import.uc_experiment_runs = 10; % univariate RNC experiment runs
cfg.sub(4).import.mc_experiment_runs = 12; % multivariate RNC experiment runs
cfg.sub(4).import.localizer = [];
cfg.sub(4).import.second_localizer = [];
cfg.sub(4).preproc.outlier_cutoff = outlier_cutoff; % first value: image difference, second value: slice difference in variance
cfg.sub(4).preproc.corrupted_volumes = []; % Numerical format: A.xy where A refers to the total volume number and xy to the slice number. Example: 768.03 means volume 768, slice 03
cfg.sub(4).preproc.corrupted_slices = [];
cfg.sub(4).preproc.eyesize_range = [80 1000 0.8];
cfg.sub(4).dicomtime = []; % we didn't record the first run

% Subject 5
cfg.sub(5).id = 'sub-05';
cfg.sub(5).nr = 5;
cfg.sub(5).age = [29];
cfg.sub(5).gender = 0; % 1 is female, 0 is male
cfg.sub(5).import.prf_experiment_runs = 3; % prf experiment runs
cfg.sub(5).import.uc_experiment_runs = 10; % univariate RNC experiment runs
cfg.sub(5).import.mc_experiment_runs = 12; % multivariate RNC experiment runs
cfg.sub(5).import.localizer = [];
cfg.sub(5).import.second_localizer = [];
cfg.sub(5).preproc.outlier_cutoff = outlier_cutoff; % first value: image difference, second value: slice difference in variance
cfg.sub(5).preproc.corrupted_volumes = []; % Numerical format: A.xy where A refers to the total volume number and xy to the slice number. Example: 768.03 means volume 768, slice 03
cfg.sub(5).preproc.corrupted_slices = [];
cfg.sub(5).preproc.eyesize_range = [80 1000 0.8];
cfg.sub(5).dicomtime = []; % we didn't record the first run

% Subject 6
cfg.sub(6).id = 'sub-06';
cfg.sub(6).nr = 6;
cfg.sub(6).age = [24];
cfg.sub(6).gender = 1; % 1 is female, 0 is male
cfg.sub(6).import.prf_experiment_runs = 3; % prf experiment runs
cfg.sub(6).import.uc_experiment_runs = 10; % univariate RNC experiment runs
cfg.sub(6).import.mc_experiment_runs = 12; % multivariate RNC experiment runs
cfg.sub(6).import.localizer = [];
cfg.sub(6).import.second_localizer = [];
cfg.sub(6).preproc.outlier_cutoff = outlier_cutoff; % first value: image difference, second value: slice difference in variance
cfg.sub(6).preproc.corrupted_volumes = []; % Numerical format: A.xy where A refers to the total volume number and xy to the slice number. Example: 768.03 means volume 768, slice 03
cfg.sub(6).preproc.corrupted_slices = [];
cfg.sub(6).preproc.eyesize_range = [80 1000 0.8];
cfg.sub(6).dicomtime = []; % we didn't record the first run


% Here, run numbers are generated from imported numbers (e.g. imported run
% 3 would be run 1, because e.g imported run 1 is brain localizer)

% Set for all subject to use all experimental runs provided
for i_sub = 1:length(cfg.sub) %cfg.subject_indices
    
    %cfg.sub(i_sub).behav_resname = sprintf('^daw_%3i.*',cfg.sub(i_sub).nr); % regexp for results file
    cfg.sub(i_sub).n_runs = cfg.sub(i_sub).import.prf_experiment_runs + ...
		cfg.sub(i_sub).import.uc_experiment_runs + ...
		cfg.sub(i_sub).import.mc_experiment_runs;
    cfg.sub(i_sub).run_indices = 1:cfg.sub(i_sub).n_runs;
end


% Run numbers to use (although we import all runs, we might have one
% corrupted run, so we might not use them all)

% If necessary, set exceptions
% example:
% cfg.sub(7).n_runs = length(cfg.sub(7).import.experiment) -1;
% cfg.sub(7).run_indices = [1:8 10];

% Subjects to be used (some subjects might have to be excluded later)
if ~isfield(cfg,'subject_indices')
    cfg.subject_indices = setdiff(1:length(cfg.sub),excluded_subjects);
end

%% Set paths    
for i = 1:length(cfg.sub)
    cfg.sub(i).dir = fullfile(derived_dir,sprintf('sub%02d',i));
end

