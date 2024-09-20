% create onset files for SPM data analysis for the study objdraw
% 
% Function for writing the onsets files for one subject - the following is
% an example of how onset files are set up to fit the needs of SPM
%
%%% Format must be : file with 3 variables - names, onset, duration

% Example for file with 3 condtions (2 task, 1 baseline) 

% names = cell(1,3);
% onsets = cell(1,3);
% durations = cell(1,3);
% 
% names{1} = 'add';
% onsets{1} = [166.7742 172.6876 177.7342 184.1456 201.53 209.7 215.9406 225.6018 320.6838 327.4248 332.6936 337.2923];
% durations{1} = [2.3868 3.0553 2.4931 2.4683 2.3368 3.5722 3.0435 3.5707 2.68 3.1327 2.2381 2.3958];
% 
% names{2} = 'sub';
% onsets{2} = [69.1668 74.833 92.3837 98.9326 104.372 111.3265 128.8686 136.7567 141.3543 244.29 250.9498 255.3798 260.4096 280.6257 288.6493 296.5246 302.4964];
% durations{2} = [2.5271 2.9987 2.3419 3.0142 1.9604 1.8954 2.7807 2.227 3.3609 1.9426 2.2744 2.4641 2.8919 2.6118 3.0432 3.2123 3.4519];
% 
% names{3} = 'fix';
% onsets{3} = [8.2313];
% durations{3} = [24.0826];
%
% Inputs: 
% cfg - cfg structure obtained from config_subjects_objdraw
% i_sub - index of the current subject 
% type - can be either "exact" or "round" -> exact means that the onsets
% are taken from the results files which are the timestamp given by
% psychtooolbox, round means the onsets are taken from the design file and
% correspond to when stimuli should have been presented - differences
% between the two timestamps are on average ~5ms 
%

function [design_mat,conditions] = create_design_from_onsets(cfg, i_sub,type)

% setup some variables 

subs = dir(fullfile('/data/pt_02348/objdraw/results/mat','OD*'));
subs = {subs.name}';
exclude_subs = 'OD03';
no_runs = 12;
no_trials = 120;
n_scans = 251; %number of volumes per block
n_cat = 48; % number of categories in the experiment 
TR = 1.5; % TR in seconds
conds = {'Photo', 'Drawing', 'Sketch'};

% first load the results file for the corresponding subject 
sub_id = subs{i_sub};

design_fp = fullfile('/data/pt_02348/objdraw/results/mat',sub_id);

% initialize design cell array

design_mat = cell(1,no_runs); 

for this_run = 1: no_runs
    
load(fullfile(design_fp,['run', num2str(this_run,'%02.f'), '.mat']));

this_design = zeros(n_scans,n_cat); 

for cat_idx = 1:n_cat
    this_onsets = [];
    for this_trial = 1:	no_trials
        if ~strcmp(results.trial(this_trial).trial_type, 'catch') && results.trial(this_trial).category_nr == cat_idx
            if strcmpi(type, 'exact')
                this_onsets =  [this_onsets,results.trial(this_trial).image_on];
            elseif strcmpi(type, 'round')
                this_onsets =  [this_onsets,results.trial(this_trial).onset];
            end
        end
    end 
    this_onsets = this_onsets/TR; 
    this_design(this_onsets,cat_idx)= 1; 
end 

if design.cond_order(this_run) ==1 
    
    this_design = cat(2,this_design,zeros(n_scans,n_cat*2)); 
    
elseif design.cond_order(this_run) ==2
    
    this_design = cat(2,zeros(n_scans,n_cat),this_design,zeros(n_scans,n_cat)); 
    
else 
    
    this_design = cat(2,zeros(n_scans,n_cat*2),this_design); 
end 

design_mat(this_run) = {this_design}; 
conditions = design.cond_order;

end 
end 