%%%%%%%%%%%%%%%%%%%%%%%%%% EXPERIMENT PARAMETERS (edit as necessary)

clear
clc

% display
ptres = [1920 1080 60 32];  % display resolution. [] means to use current display resolution. Set to [1920 1080 60 32] for the CCNB scanner.

% fixation dot
fixationinfo = {uint8([255 0 0; 0 0 0; 255 255 255]) 0.5};  % dot colors and alpha value
fixationsize = 10;          % dot size in pixels
meanchange = 3;            % dot changes occur with this average interval (in seconds)
changeplusminus = 2;       % plus or minus this amount (in seconds)

% trigger
triggerkey = '5'; % stimulus starts when this key is detected --> Change to scanner trigger. Set to '5' for the CCNB scanner
tfun = @() fprintf('STIMULUS STARTED.\n');  % function to call once trigger is detected

% tweaking
offset = [0 0];            % [X Y] where X and Y are the horizontal and vertical
                           % offsets to apply.  for example, [5 -10] means shift 
                           % 5 pixels to right, shift 10 pixels up.
movieflip = [0 0];         % [A B] where A==1 means to flip vertical dimension
                           % and B==1 means to flip horizontal dimension

% Directories
stimulusdir = '../06_in_vivo_validation/01_experimental_paradigm/prf_experiment/prf_stimuli';
knkutils_dir = '../06_in_vivo_validation/01_experimental_paradigm/prf_experiment/knkutils-master';
save_dir_parent = '../relational_neural_control/in_vivo_validation/fmri_dataset/beh';

% Add utility functions to path
addpath(genpath(knkutils_dir))

%%%%%%%%%%%%%%%%%%%%%%%%%% DO NOT EDIT BELOW

% set rand state
rand('state',sum(100*clock));
randn('state',sum(100*clock));

% ask the user what to run
if ~exist('subjnum','var') 
  subjnum = input('What is the subj id? ')
end
expnum = input('What experiment (89=CCW, 90=CW, 91=expand, 92=contract, 93=multibar, 94=wedgeringmash)? ')
runnum = input('What run number (for filename)? ')

% Create saving directory if not existing
if isunix == 1
	slash = '/'; % directory slash for linux/max
else
	slash = '\'; % directory slash for windows
end
save_dir = [save_dir_parent, slash, 'sub-', sprintf('%02d', subjnum), ...
	'prf_experiment']
if ~exist(save_dir)
	mkdir(save_dir)
end
cd(save_dir)

% prepare inputs
trialparams = [];
ptonparams = {ptres,[],0};
dres = [];
frameduration = 4;
grayval = uint8(127);
iscolor = 1;
soafun = @() round(meanchange*(60/frameduration) + changeplusminus*(2*(rand-.5))*(60/frameduration));

% load specialoverlay
a1 = load(fullfile(stimulusdir,'fixationgrid.mat'));

% some prep
if ~exist('images','var')
  images = [];
  maskimages = [];
end
filename = sprintf('%s_subj%d_run%02d_exp%02d.mat',gettimestring,subjnum,runnum,expnum);

% run experiment
[images,maskimages] = ...
  showmulticlass(filename,offset,movieflip,frameduration,fixationinfo,fixationsize,tfun, ...
                 ptonparams,soafun,0,images,expnum,[],grayval,iscolor,[],[],[],dres,triggerkey, ...
                 [],trialparams,[],maskimages,a1.specialoverlay,stimulusdir);

%%%%%%%%%%%%%%%%%%%%%%%%%%

% KK notes:
% - remove performance check at end
% - remove resampling and viewingdistance stuff
% - remove hresgrid and vresgrid stuff
% - hardcode grid and pregenerate
% - trialparams became an internal constant
