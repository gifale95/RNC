function aperture = stimuluspreprocessing()
% this is an example of pre-processing of the stimulus masks.
% this script requires some utilities from http://github.com/kendrickkay/knkutils/

% load the stimuli
a1 = load('stimuli.mat');

% Add utility functions to path
knkutils_dir = '/home/ale/aaa_stuff/PhD/projects/mid_level_vision/code/new_data_collection/02_preprocessing/fmri/02_prf_mapping/get_aperture/knkutils-master';
addpath(genpath(knkutils_dir))

% define the resolution that we want to use
res = 200;

% resize the aperture masks and change the range to [0,1]
stim = zeros(res,res,size(a1.masks,3));
for p=1:size(a1.masks,3), p
  stim(:,:,p) = normalizerange(imresize(double(a1.masks(:,:,p))/255,[res res],'cubic'),0,1,0,1);
end

% add on a blank image and make the 0s in the indices refer to this blank image
stim(:,:,end+1) = 0;
a1.multibarindices(a1.multibarindices==0)   = size(stim,3);
a1.wedgeringindices(a1.wedgeringindices==0) = size(stim,3);

% use the indices to construct the entire stimulus movie for both types of runs
stim = {stim(:,:,a1.multibarindices) stim(:,:,a1.wedgeringindices)};

% to reduce computational time, we will average over successive chunks of the movie
% so that we have fewer stimulus frames to deal with.  here we average over chunks
% of size 15 since the stimulus movie is at 15 Hz.
for p=1:length(stim)
  stim{p} = blob(stim{p},3,15)/15;
end

% % check the dimensionality
% stim

% % save the result so you don't have to compute this again
% save('stimuli_preprocessed.mat','stim');

aperture = stim{1,1}; % only save the bar stimulus
save('stimulus_bar.mat','aperture')

end
