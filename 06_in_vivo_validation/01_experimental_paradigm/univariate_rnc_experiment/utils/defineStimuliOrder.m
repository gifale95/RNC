function defineStimuliOrder(data, stim_order_dir, slash)
% This function generates the pseudo-randomized stimuli order.

	% Define the random image order, making sure that no image is repeated in
	% two consecutive trials
	all_images = 1:data.paradigm.tot_images;
	image_order = [];
	for r = 1:data.paradigm.image_repeats
		ordered_trials = Shuffle(all_images);
		if r ~= 1
			while any(ismember(image_order(end-5:end), ordered_trials(1:5)))
				ordered_trials = Shuffle(all_images);
			end
		end
		image_order = [image_order, ordered_trials];
	end

	% Define the blank trials in each run
	blank_trials = [];
	for r = 1:data.paradigm.total_runs
		blank_trials_run = [];
		% Blank trials at beginning of runs
		blank_trials_run = [blank_trials_run, 1:data.paradigm.blank_trials_run_start];
		% Blank trials in the middle of runs
		part_trials = data.paradigm.trials_per_run - ...
			data.paradigm.blank_trials_run_start - ...
			data.paradigm.blank_trials_run_end;
		step = floor(part_trials / (data.paradigm.blank_trials_run_middle));
		idx_1 = data.paradigm.blank_trials_run_start + step;
		idx_2 = data.paradigm.blank_trials_run_middle * (step + 1);
		trials = idx_1:step:idx_2;
		noise = randi([-3 2], 1, data.paradigm.blank_trials_run_middle);
		blank_trials_run = [blank_trials_run, trials+noise];
		% Blank trials at end of runs
		idx_1 = data.paradigm.trials_per_run - ...
			data.paradigm.blank_trials_run_end + 1;
		idx_2 = data.paradigm.trials_per_run;
		blank_trials_run = [blank_trials_run, idx_1:idx_2];
		blank_trials = [blank_trials; blank_trials_run];
	end
	
	% Define the catch trials in each run
	catch_trials = [];
	for r = 1:data.paradigm.total_runs
			catch_trials_run = [1];
		while any(ismember(blank_trials(r,:), catch_trials_run))
			catch_trials_run = 1:data.paradigm.trials_per_run;
			catch_trials_run = Shuffle(catch_trials_run);
			catch_trials_run = catch_trials_run(1:data.paradigm.catch_trials_per_run);
		end
		catch_trials = [catch_trials; catch_trials_run];
	end
	
	% Create the 2-D stimuli presentation array of shape: (total_runs × Trials per run)
	% Each entry indicates the corresponding image number to present
	stim_order = zeros([data.paradigm.total_runs, data.paradigm.trials_per_run]);
	counter = 1;
	for r = 1:data.paradigm.total_runs
		if any(ismember(blank_trials(r,:), catch_trials(r,:)))
			error('Blank and catch trials overlap!')
		end
		for t = 1:data.paradigm.trials_per_run
			if any(ismember(blank_trials(r,:), t))
				stim_order(r,t) = 0;
			elseif any(ismember(catch_trials(r,:), t))
				stim_order(r,t) = 9999;
			else
				stim_order(r,t) = image_order(counter);
				counter = counter + 1;
			end
		end
	end

	% Save the stimuli order
	save([stim_order_dir, slash, 'stim_order_sub-', ...
		sprintf('%02d', data.subject.id)], 'stim_order')
	
end
