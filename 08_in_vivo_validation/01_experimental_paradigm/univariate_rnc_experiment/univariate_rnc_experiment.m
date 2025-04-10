% Univariate RNC images
% V1-V4 ROI pair
% Univariate RNC controlling image conditions:
	% 25 high-V1_high-hV4 images (from NSD)
	% 25 low-low-hV4 images (from NSD)
	% 25 high-low-hV4 images (from NSD)
	% 25 low-V1_high-hV4 images (from NSD)
	% 25 V1 baseline images (from NSD)
	% 25 hV4 baseline images (from NSD)
% Paradigm:
	% 2000ms image onscreen + 2000ms blank screen
	% Target detection task (press button if Buzz Lightyear appears)
	% 900 stimuli trials (6 repeats for each of the 150 images)

% This code is available at:
% https://github.com/gifale95/RNC/blob/main/06_in_vivo_validation/01_experimental_paradigm/univariate_rnc_experiment/univariate_rnc_experiment.m


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PTB checks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
clc

rng('shuffle'); % randomization of trials according to Matlab internal time
KbName('UnifyKeyNames') % (following MacOS-X naming scheme)
Priority(2); % 2 (setting high-priority for Psychtoolbox timing)

% Get rid of warning messages (and other Psychtoolbox stuff)
Screen('Preference', 'SkipSyncTests', 0); %%% 0
Screen('Preference', 'VisualDebugLevel', 2); %%% 2
Screen('Preference', 'SuppressAllWarnings', 0); %%% 0
KbQueueRelease; % in case we still have a queue running



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To edit
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MRI related
mri_trigger_wait = 1; % Set to 1 to wait for MRI trigger

% Directories
stimuli_dir = '../stimuli/univariate_rnc_controlling_images/V1-V4/imageset-nsd'; % Download from: https://openneuro.org/datasets/ds005503
catch_stimuli_dir = '../stimuli/catch_images'; % Download from: https://openneuro.org/datasets/ds005503
utils_dir = '../06_in_vivo_validation/01_experimental_paradigm/univariate_rnc_experiment/utils';
stim_order_dir = '../06_in_vivo_validation/01_experimental_paradigm/univariate_rnc_experiment/stimuli_order';
save_dir = '../relational_neural_control/in_vivo_validation/beh';

% Select slash type and add utils path
if isunix == 1
	slash = '/'; % directory slash for linux/max
else
	slash = '\'; % directory slash for windows
end
addpath(utils_dir);

% Stimuli size
data.stim_size.pix_per_deg = 71;
data.stim_size.vis_angle_x = 8.4;
data.stim_size.vis_angle_y = 8.4;
data.stim_size.vis_angle_fixation_dot = 0.2;
data.stim_size.image_size_pixels_x = round(data.stim_size.vis_angle_x * data.stim_size.pix_per_deg);
data.stim_size.image_size_pixels_y = round(data.stim_size.vis_angle_y * data.stim_size.pix_per_deg);
data.stim_size.fixation_dot_size_pixels = round(data.stim_size.vis_angle_fixation_dot * data.stim_size.pix_per_deg);

% Screen window number for PTB 'OpenWindow' function
screenWin = 0;

% Milliseconds to be subtracted from the flipping time so to not miss the
% visual frame (and having to wait an additional 16ms (or more, or less,
% depending on the display device refresh rate) for the flip).
% If not needed, set to 0
fixFlipTime=.01;

% Name of response keys
quit_key = KbName('q'); % press (during image presentation) to exit the experiment
response_keys = [KbName('1!') KbName('2@'), KbName('3#'), KbName('4$')];

% Scanner trigger
scanner_trigger = KbName('5%');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Experimental parameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Timing
data.paradigm.img_duration = 2; % 2s of image onscreen duration
data.paradigm.isi = 2; % 2s of inter-stimulus interval
data.paradigm.iti = data.paradigm.img_duration + data.paradigm.isi;

% Presentation structure
data.paradigm.stimuli_trials_per_run = 90;
data.paradigm.catch_trials_per_run = 4;
data.paradigm.blank_trials_run_start = 3;
data.paradigm.blank_trials_run_middle = 8;
data.paradigm.blank_trials_run_end = 4;
data.paradigm.trials_per_run = data.paradigm.stimuli_trials_per_run + ...
	data.paradigm.catch_trials_per_run + ...
	data.paradigm.blank_trials_run_start + ...
	data.paradigm.blank_trials_run_middle + ...
	data.paradigm.blank_trials_run_end; % 436s per run (~7m)
data.paradigm.total_runs = 10; % total of ~72m + pauses

% Stimuli
data.paradigm.tot_images = 150;
data.paradigm.image_repeats = 6;
data.paradigm.catch_images = 10;
data.paradigm.background_color = [127 127 127];

% Experiment_time
exptime = datestr(now,'yyyy-mm-dd_HH-MM-SS');
data.experiment_time = exptime;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Subject's info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data.subject.id = input('Subject''s Number: --> ');
data.subject.age = input('Subject''s Age: --> ');
data.subject.sex = input('Subject''s Sex: --> ','s');
data.run = input('Run: --> ');

% Create saving directory if not existing
output_save_dir = [save_dir, slash, 'sub-', ...
	sprintf('%02d', data.subject.id), 'univariate_rnc_experiment'];
if ~exist(output_save_dir)
	mkdir(output_save_dir)
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read the stimuli into a structure
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Create a structure with the stimuli images
stimuli = getStimuli(stimuli_dir, slash);

% Create a structure with the catch images
catch_stimuli = getCatchStimuli(catch_stimuli_dir, slash);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the presentation order
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function generates the pseudo-randomized stimuli order
%defineStimuliOrder(data, stim_order_dir, slash);

% Load the (already defined) stimuli and task order
load([stim_order_dir, slash, 'stim_order_sub-', sprintf('%02d', data.subject.id)])
data.presentation_order = stim_order;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define the trials onset and offset timings
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

trial_onsets = 0:data.paradigm.iti:data.paradigm.trials_per_run*data.paradigm.iti;
trial_offsets = trial_onsets + data.paradigm.img_duration;



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Open Psychtoolbox
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[win, screenRect] = Screen('OpenWindow', screenWin, ...
	data.paradigm.background_color);
Screen('BlendFunction', win, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
%VBLSyncTest() %%%

HideCursor(0)
Screen('TextSize', win, floor(screenRect(3) / 50));
Screen('TextStyle', win, 0);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Define coordinates of the destination rectangle
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

destRect = [(screenRect(3) / 2) - floor(data.stim_size.image_size_pixels_x / 2), ...
	(screenRect(4) / 2) - floor(data.stim_size.image_size_pixels_y / 2), ...
	(screenRect(3) / 2) + floor(data.stim_size.image_size_pixels_x / 2), ...
    (screenRect(4) / 2) + floor(data.stim_size.image_size_pixels_y / 2)];



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display instructions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Instructions
text = strcat(' RUN %d/%d\n\n\n', ...
	'Always gaze at the central fixation dot.\n\n', ...
	'Focus and pay attention to the images.\n\n', ...
	'Press the button if Buzz Lightyear appears.\n\n');
instructions = sprintf(text, data.run, data.paradigm.total_runs);
DrawFormattedText(win, instructions, 'center', 'center', [0 0 0]);

% Fixation dot
fixationDot(win, screenRect, data.stim_size.fixation_dot_size_pixels);

% Flip screen
Screen('Flip', win);


 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 % Wait for MRI trigger
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

KbQueueCreate;
KbQueueStart;
ListenChar;

% The MRI trigger is sent at the beginning of the MRI sequence
if mri_trigger_wait == 1
	[~, keyCode] = KbQueueCheck;
	disp('waiting for trigger');
	while sum(keyCode(scanner_trigger)) == 0
		[~, keyCode] = KbQueueCheck;
		WaitSecs('Yieldsecs', 0.001);
	end
end
 
data.run_start = GetSecs;


 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image presentation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Pre-run baseline with fixation dot
fixationDot(win, screenRect, data.stim_size.fixation_dot_size_pixels);
t2 = Screen('Flip', win);

data.trials = [];
data.task = [];

for trial = 1:data.paradigm.trials_per_run

	% Reset button presses
	all_key_id = [];
	all_key_time = [];

	% Print trial
	fprintf('********************************************************\n');
	fprintf('********** SUB %02i, RUN %02i/%02i,  TRIAL %03i/%03i  **********\n', ...
		data.subject.id, data.run, data.paradigm.total_runs, trial, ...
		data.paradigm.trials_per_run);
	fprintf('********************************************************\n');

	% Define image onset time
	onset_time = data.run_start + trial_onsets(trial) - fixFlipTime;
	 
	% Present the image
	if stim_order(data.run,trial) == 0 % blank screen trials
		tex = Screen('MakeTexture', win, data.paradigm.background_color);
	elseif stim_order(data.run,trial) == 9999 % catch trials
		catch_rand_idx = randi(data.paradigm.catch_images);
		tex = Screen('MakeTexture', win, catch_stimuli(catch_rand_idx).image);
	else % stimulus trials
		tex = Screen('MakeTexture', win, stimuli(stim_order(data.run,trial)).image);
	end
	Screen('DrawTexture', win, tex, [], destRect);
	fixationDot(win, screenRect, data.stim_size.fixation_dot_size_pixels)
	t1 = Screen('Flip', win, onset_time);
	 
	% Define image offset time
	offset_time = data.run_start + trial_offsets(trial) - fixFlipTime;
	
	% Collect all button presses during image onset (excluding last 2ms)
	[key_id, key_time] = wait_and_get_keys(offset_time-0.002, ...
		[response_keys quit_key]);
	if any(key_id == quit_key)
		disp('Quit key pressed.')
		KbQueueRelease; % release keyboard queue
		Screen('CloseAll')
		ListenChar(0);
		return
	end
	if ismember(response_keys, key_id)
		key_time = key_time(key_id ~= scanner_trigger);
		key_id = key_id(key_id ~= scanner_trigger);
	end
	all_key_id = [all_key_id key_id];
	all_key_time = [all_key_time key_time];

	% Inter-image blank screen with fixation dot
	fixationDot(win, screenRect, data.stim_size.fixation_dot_size_pixels)
	t2 = Screen('Flip', win, offset_time);

	% Close the texture after each trial (to save computation time)
	Screen('Close', tex)

	% Collect all button presses after image offset (excluding last 100 ms of trial)
	[key_id, key_time] = wait_and_get_keys(offset_time+data.paradigm.isi-0.1, ...
		[response_keys quit_key]);
	if any(key_id == quit_key)
		disp('Quit key pressed.')
		KbQueueRelease; % release keyboard queue
		Screen('CloseAll')
		ListenChar(0);
		return
	end         
	if ismember(response_keys, key_id)
		key_time = key_time(key_id ~= scanner_trigger);
		key_id = key_id(key_id ~= scanner_trigger);
	end
	all_key_id = [all_key_id key_id];
	all_key_time = [all_key_time key_time];
	
	% Save the trial info into the results structure
	data.trials(trial).run = data.run;
	data.trials(trial).trial = trial;
	data.trials(trial).absolute_img_onset = t1;
	data.trials(trial).absolute_img_offset = t2;
	data.trials(trial).relative_img_onset = t1 - data.run_start;
	data.trials(trial).relative_img_offset = t2 - data.run_start;
	data.trials(trial).img_duration = t2 - t1;
	if stim_order(data.run,trial) == 0 % blank trials
		data.trials(trial).image_number = NaN;
		data.trials(trial).image_type = 'blank';
		data.trials(trial).image_name = NaN;
	elseif stim_order(data.run,trial) == 9999 % catch trials
		data.trials(trial).image_number = ...
			catch_stimuli(catch_rand_idx).image_number;
		data.trials(trial).image_type = catch_stimuli(catch_rand_idx).image_type;
		data.trials(trial).image_name = catch_stimuli(catch_rand_idx).name;
	else % stimuli trials
		data.trials(trial).image_number = ...
			stimuli(stim_order(data.run, trial)).image_number;
		data.trials(trial).image_type = ...
			stimuli(stim_order(data.run, trial)).image_type;
		data.trials(trial).image_name = ...
			stimuli(stim_order(data.run, trial)).name;
	end
	
	% Save the button responses
	data.trials(trial).responded = false;
	button_pressed = ~isempty(all_key_id(all_key_id ~= scanner_trigger));
	try % add a try-catch around to prevent any odd things from throwing an error
		if button_pressed
			data.trials(trial).responded = true;
			data.trials(trial).key_id = all_key_id(all_key_id ~= scanner_trigger); 
			data.trials(trial).key_time = all_key_time;
		end
	catch
		data.trials(trial).key_id = 'something weird happened.';
		disp('something weird happened with recording button presses');
		disp(['all_key_id = ' num2str(all_key_id)])
	end
	
	% Print behavioral performance of each trial
	if strcmp(data.trials(trial).image_type, 'catch')
		if button_pressed
			disp('BUTTON PRESSED, HIT.');
		else
			disp('BUTTON NOT PRESSED, MISS.');
		end
	else
		if button_pressed
			disp('BUTTON PRESSED, FALSE ALARM.')
		end
			% there should be a lot of correct rejections, so let's ignore those.
	end

end % trial



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Display behavioral performance of the entire run
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try
	catch_trials = strcmp({data.trials.image_type}, 'catch');
	noncatch_trials = ~strcmp({data.trials.image_type}, 'catch');
	hit_rate = 100 * mean([data.trials(catch_trials).responded]);
	fa_rate = 100 * mean([data.trials(noncatch_trials).responded]);
	fprintf('Hit rate: %.2f%%\n', hit_rate)
	fprintf('False Alarm rate (incl. long latency hits): %.2f%%\n', fa_rate)
catch
	disp('could not calculate performance online.')
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save the data structure after each run
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

save([output_save_dir, slash, 'sub-', sprintf('%03d', data.subject.id), ...
	'_run-', sprintf('%03d', data.run), '_', exptime, '.mat'], 'data')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exit experiment
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

KbQueueStop;
Screen('CloseAll')
ListenChar(0);



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [key_id, key_time] = wait_and_get_keys(until, key_codes)
	% each button may only be pressed once per call
	key_id = []; key_time = [];
	while GetSecs < until
		[pressed, keymap] = KbQueueCheck;
		keys = keymap(key_codes);
		if pressed && any(keys)
			code = key_codes(find(keys, 1, 'first'));
			secs = keys(find(keys, 1, 'first'));
			if any(key_id == code), continue, end
			key_id = [key_id code];
			key_time = [key_time secs];
		end
		WaitSecs('Yieldsecs', 0.001);
	end
end


