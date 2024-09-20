%% control analysis - check if subjects pressed buttons and if there are subjects which missed a lot of buttons 

clc 
clear all
% setup some variables 

subs = dir(fullfile('/data/pt_02348/objdraw/results/mat','OD*'));
subs = {subs.name}';
excluded_subs = {'od03'};%,'od12','od13','od23','od29'};
no_runs = 12;
no_trials = 120;
n_cat = 48; % number of categories in the experiment 
n_rep = 2; %number of repetitions of one category per block

button_press_all_subs = [];

for sub = 1:length(subs)

% first load the results file for the corresponding subject 
sub_id = subs{sub};

if any(strcmpi(excluded_subs, sub_id)), continue, end 

design_fp = fullfile('/data/pt_02348/objdraw/results/mat',sub_id);

for this_run = 1: no_runs
    
load(fullfile(design_fp,['run', num2str(this_run,'%02.f'), '.mat']));

responded = 0;

for this_trial = 1:length(results.trial)
    if strcmpi(results.trial(this_trial).trial_type, 'catch') && results.trial(this_trial).responded
        responded = responded+1; 
    end 
end 

button_press_all_subs(sub,this_run) = responded; 

end 
end 

%% plot some summary statistics 

sub_ids = setdiff(lower(subs),excluded_subs); 

figure
bar(mean(button_press_all_subs,2)/24);
xticks(1:31)
title('Mean Percentage Hits in Catch Trials across subjects');

figure 
subplot(2,2,1)
bar(button_press_all_subs(7,:)/24);
title('HITs for sub 7')
subplot(2,2,2)
bar(button_press_all_subs(17,:)/24);
title('HITs for sub 17')
subplot(2,2,3)
bar(button_press_all_subs(22,:)/24);
title('HITs for sub 22')
subplot(2,2,4)
bar(button_press_all_subs(31,:)/24);
title('HITs for sub 31')

excluded_subs = mean(button_press_all_subs,2)/24 < 0.8;