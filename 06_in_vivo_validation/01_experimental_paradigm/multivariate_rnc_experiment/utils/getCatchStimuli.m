function stimuli = getCatchStimuli(catch_stimuli_dir, slash)
% This function generates a structure with the stimuli images.

	% Get images file names
	image_files = dir(catch_stimuli_dir);
	image_files = image_files(3:end);

	for i = 1:length(image_files)

		% Image number
		stimuli(i).image_number = i;
		
		% Image type
		stimuli(i).image_type = 'catch';

		% Image file name
		stimuli(i).name = image_files(i).name;
		% Video folder
		stimuli(i).folder = image_files(i).folder;

		% Load the images
		img=imread([image_files(i).folder, slash, image_files(i).name]);
		stimuli(i).image = img;

	end

end