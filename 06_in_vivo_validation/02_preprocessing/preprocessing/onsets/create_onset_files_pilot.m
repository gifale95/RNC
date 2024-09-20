% create onset files for SPM data analysis for the study rcor -
% VERSION for fixing wrong image path , category number match 

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

function create_onset_files_pilot(cfg, i_sub, sub_id, type)

% setup some variables 

subs = dir(fullfile(cfg.dirs.data_dir,'results','RCOR*'));
subs = {subs.name}';
exclude_subs = 'RCOR_TEST';
%no_runs_first = length(cfg.sub(i_sub).import.experiment);
%no_runs_second = length(cfg.sub(i_sub).import.second_experiment);
no_runs = [length(cfg.sub(i_sub).import.experiment) length(cfg.sub(i_sub).import.second_experiment)];
no_runs_total = length(cfg.sub(i_sub).import.experiment) + length(cfg.sub(i_sub).import.second_experiment); 
no_trials = 151; 
n_cat = 121; % number of categories in the experiment 
n_rep = 1; %number of repetitions of one category per block


%pid = cfg.sub(i_sub).pid; 
i_sub = num2str(i_sub,'%02.f');

    
% if the folder already exists then empty it first 
if exist(fullfile(cfg.dirs.derived_dir,['sub' i_sub], 'onsets','cat_onsets'))
    files = dir(fullfile(cfg.dirs.derived_dir,['sub' i_sub], 'onsets','cat_onsets'));
    for k = 1:length(files)
        delete([fullfile(cfg.dirs.derived_dir,['sub' i_sub], 'onsets','cat_onsets') '/' files(k).name]);
    end
    rmdir(fullfile(cfg.dirs.derived_dir,['sub' i_sub], 'onsets','cat_onsets'));
    mkdir(fullfile(cfg.dirs.derived_dir,['sub' i_sub], 'onsets','cat_onsets'));
end 

run_ix = 0;

for ses = 1:2
for this_run = 1: no_runs(ses)
    run_ix = run_ix + 1;
    
load(fullfile(cfg.dirs.data_dir,'results',sub_id{ses},'fmri',['run', num2str(this_run,'%02.f'), '_fmri.mat']));

onsets = cell(1,n_cat);

names = cell(1,n_cat);  

cats = [results.trial.image_nr];
cats = cats(cats~=243);

for cat_ix = 1:length(cats)
    this_cat = cats(cat_ix);
    this_onsets = [];
    for this_trial = 1:	no_trials
        if ~strcmp(results.trial(this_trial).trial_type, 'catch') && results.trial(this_trial).category_nr == this_cat
            if strcmpi(type, 'exact')
                this_onsets =  [this_onsets,results.trial(this_trial).image_on];
            elseif strcmpi(type, 'round')
                this_onsets =  [this_onsets,results.trial(this_trial).onset];
            end
        end
    end 
    onsets(cat_ix) = {this_onsets};
    names(cat_ix) = {['Image_' num2str(this_cat)]};
    durations(cat_ix) = {repmat(0.5,1,n_rep)};
end 

% if folder does not exist then create it
if ~exist(fullfile(cfg.dirs.derived_dir,['sub' i_sub], 'onsets','cat_onsets'))
    mkdir(fullfile(cfg.dirs.derived_dir,['sub' i_sub], 'onsets','cat_onsets'));
end

save(fullfile(cfg.dirs.derived_dir,['sub' i_sub], 'onsets','cat_onsets',['cat_onsets_', 'sub', i_sub, '_run',num2str(run_ix,'%02.f'),'.mat']), 'names', 'onsets', 'durations');   

end 

end
%THIS IS FOR CONTROVERSIAL STIMULI. IF IT DOESN'T MAKE SENSE, JUST DISCARD.
% ct = 1; 
% for this_run = no_runs_first+1:no_runs_total
%     
% load(fullfile(cfg.dirs.data_dir,'results',[pid '+'],'fmri',['run', num2str(ct,'%02.f'), '_fmri.mat']));
% 
% onsets = cell(1,n_cat);
% 
% names = cell(1,1);  
% 
% all_controversial_cats = cell(1);
% 
% for this_trial = 1:	no_trials
%     
%     if ~strcmp(results.trial(this_trial).trial_type, 'catch')
%         this_cat = strsplit(results.trial(this_trial).image_path,'/');
%         all_controversial_cats = cat(1,{this_cat{3}(1:end-4)},all_controversial_cats)
%         
%     end
% end 
% 
% all_controversial_cats(121) = [];
% all_controversial_cats = unique(all_controversial_cats);
% all_controversial_cats = natsortfiles(all_controversial_cats);
% 
% for contr_cat = 1:length(all_controversial_cats)
%     
%     this_contr_cat = all_controversial_cats{contr_cat}; 
%     this_onsets= [];
%     for this_trial = 1:no_trials 
%         if ~strcmp(results.trial(this_trial).trial_type, 'catch')
%         this_cat = strsplit(results.trial(this_trial).image_path,'/');
%         if strcmpi({this_cat{3}(1:end-4)}, this_contr_cat)
%             if strcmpi(type, 'exact')
%                 this_onsets =  [this_onsets, results.trial(this_trial).image_on];
%             elseif strcmpi(type, 'round')
%                 this_onsets =  [this_onsets,  results.trial(this_trial).onset];
%             end
%         end 
%         end 
%     end 
% 
%     onsets(contr_cat) = {this_onsets};
%     names(contr_cat) = {['Controversial_Image_' num2str(contr_cat)]};
%     durations(contr_cat) = {repmat(0.5,1,n_rep)};
%     end 
% 
% save(fullfile(cfg.dirs.derived_dir,['sub' i_sub], 'onsets','cat_onsets',['cat_onsets_', 'sub', i_sub, '_run',num2str(this_run,'%02.f'),'.mat']), 'names', 'onsets', 'durations');   
% 
% ct = ct+1;
% end 

end 