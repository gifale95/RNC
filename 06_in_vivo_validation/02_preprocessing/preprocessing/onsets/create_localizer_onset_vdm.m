% create localizer onsets 
function create_localizer_onset_vdm(cfg,i_sub,loc_count)

for i = 1:loc_count
% setup some variables 

data_dir = fullfile(cfg.dirs.data_dir, 'results',cfg.sub(i_sub).pid{i});
exclude_subs = '';
load(fullfile(data_dir, 'localizer_results.mat')); 
sequence = loc_results.sequence; 
%fixation_duration = 7.5;
%stimulation_duration = 15;

%sub_id =  cfg.sub(i_sub).pid;
onsets = cell(1,3);

onset = [];

for run = 1:length(sequence) 
    if sequence(run)== 1; onset = [onset sum(sequence(1:run-1) ==0)*7.5+sum(sequence(1:run-1)>0)*15]; end; 
end 

onsets(1,1) = {onset};

onset=[];

for run = 1:length(sequence)
    if sequence(run) == 2; onset = [onset sum(sequence(1:run-1) ==0)*7.5+sum(sequence(1:run-1)>0)*15]; end; 
end 

onsets(1,2) = {onset};

onset = []; 

for run = 1:length(sequence) 
    if sequence(run)== 3; onset = [onset sum(sequence(1:run-1) ==0)*7.5+sum(sequence(1:run-1)>0)*15]; end; 
end 

onsets(1,3) = {onset};

names= cell(1,3);

names{1} = 'objects';
names{2} = 'scrambled';
names{3} = 'scenes';

durations = cell(1,3);

durations{1} = repmat(15,1,8);
durations{2} = repmat(15,1,8);
durations{3} = repmat(15,1,8);


if ~exist(fullfile(cfg.dirs.derived_dir,sprintf('sub%02d',i_sub), 'onsets','localizer'))
mkdir(fullfile(cfg.dirs.derived_dir,sprintf('sub%02d',i_sub), 'onsets','localizer'));
end 
if i ==1 % for first localizer use different name than for second
    save(fullfile(cfg.dirs.derived_dir,sprintf('sub%02d',i_sub), 'onsets','localizer','onsets_localizer'), 'onsets', 'names', 'durations')
elseif i ==2 
    save(fullfile(cfg.dirs.derived_dir,sprintf('sub%02d',i_sub), 'onsets','localizer','onsets_localizer_second'), 'onsets', 'names', 'durations')
end
end 
end 