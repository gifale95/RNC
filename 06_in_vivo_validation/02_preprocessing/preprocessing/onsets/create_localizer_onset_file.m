% create localizer onsets 

clear all

% setup some variables 

subs = dir(fullfile('/data/pt_02348/objdraw/results/mat','VDM*'));
subs = {subs.name}';
exclude_subs = '';
sequence = [0 1 2 0 2 1 0 2 1 0 1 2 0 1 2 0 2 1 0 2 1 0 1 2 0 1 2 0 2 1 0 2 1 0 1 2 0 0];
fixation_duration = 7.5;
stimulation_duration = 15;

for sub = 12%:length(subs)
% first load the results file for the corresponding subject 
sub_id = subs{sub};

if sub_id == exclude_subs, continue, end 

onset = [];

for run = 1:length(sequence)-1 
    if sequence(run)== 0 | sequence(run) ==2 continue; end; 
    onset = [onset sum(sequence(1:run-1) ==0)*7.5+sum(sequence(1:run-1)==1)*15+sum(sequence(1:run-1)==2)*15];
end 

onsets(1,1) = {onset};

onset=[];

for run = 1:length(sequence)-1 
    if sequence(run)== 0 | sequence(run) ==1 continue; end; 
    onset = [onset sum(sequence(1:run-1) ==0)*7.5+sum(sequence(1:run-1)==1)*15+sum(sequence(1:run-1)==2)*15];
end 

onsets(1,2) = {onset};

names= cell(1,2);

names{1} = 'object';
names{2} = 'scrambled';

durations = cell(1,2);

durations{1} = repmat(15,1,12);
durations{2} = repmat(15,1,12);


if ~exist(fullfile('/data/pt_02350/derived',['sub' sub_id(3:end)], 'onsets','localizer'))
mkdir(fullfile('/data/pt_02350/derived',['sub' sub_id(3:end)], 'onsets','localizer'));
end 
save(fullfile('/data/pt_02350/derived',['sub' sub_id(3:end)], 'onsets','localizer','onsets_localizer'), 'onsets', 'names', 'durations')
end 