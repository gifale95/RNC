%% fMRI RSA-ROI analysis wrapper

clear all
clc

%setup paths

fmri_path = '/Users/johannessinger/scratch/rcor_collab/derived';
% set figurepath 
figure_path = '/Users/johannessinger/scratch/rcor_collab/derived/group';

% get fmri subnames 

fmri_subs = dir(fullfile(fmri_path,'*sub*'));
fmri_subs = {fmri_subs.name}';

% specify excluded subjects

excluded_subs = {};

% initialize fMRI RDMs
control_RDMs = [];
challenge_RDMs = [];

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    this_control_RDM = [];
    this_challenge_RDM = [];
    
    if any(ismember(excluded_subs,sub_id)), continue, end,
    
    % load fMRI RDMs 
    
    load(fullfile(fmri_path,[sub_id], 'results','RSA_hrf_fitting','roi', 'control','res_other_average.mat'));
    
    for roi=1:length(results.other_average.output)
        this_control_RDM = cat(3, this_control_RDM, 1-results.other_average.output{roi});
    end
    
    control_RDMs = cat(4,control_RDMs, this_control_RDM);
    
        load(fullfile(fmri_path,[sub_id], 'results','RSA_hrf_fitting','roi', 'challenge','res_other_average.mat'));
    
    for roi=1:length(results.other_average.output)
        this_challenge_RDM = cat(3, this_challenge_RDM, 1-results.other_average.output{roi});
    end
    
    challenge_RDMs = cat(4,challenge_RDMs, this_challenge_RDM);
    
end

%% plot mean RDMs 

figure

for roi = 1:size(control_RDMs,3)
    
    subplot(1,3,roi)
    
    imagesc(squareform(tiedrank(squareform(squeeze(mean(control_RDMs(:,:,roi,:),4)))))); 
    axis square
end 


figure

for roi = 1:size(control_RDMs,3)
    
    subplot(1,3,roi)
    
    imagesc(squareform(tiedrank(squareform(squeeze(mean(challenge_RDMs(:,:,roi,:),4)))))); 
    axis square
end 

%% save RDMs 

mean_EVC_RDM_control = squeeze(mean(control_RDMs(:,:,1,:),4));

mean_PPA_RDM_control = squeeze(mean(control_RDMs(:,:,2,:),4));

mean_LOC_RDM_control = squeeze(mean(control_RDMs(:,:,3,:),4));

save(fullfile(figure_path,'mean_RDMs_control.mat'),'mean_EVC_RDM_control', 'mean_LOC_RDM_control','mean_PPA_RDM_control')

mean_EVC_RDM_challenge = squeeze(mean(challenge_RDMs(:,:,1,:),4));

mean_PPA_RDM_challenge = squeeze(mean(challenge_RDMs(:,:,2,:),4));

mean_LOC_RDM_challenge = squeeze(mean(challenge_RDMs(:,:,3,:),4));

save(fullfile(figure_path,'mean_RDMs_challenge.mat'),'mean_EVC_RDM_challenge', 'mean_PPA_RDM_challenge','mean_LOC_RDM_challenge')

%% correlate challenge and control RDMs 

for sub = 1:size(control_RDMs,4)
    for roi = 1:size(control_RDMs,3)
    
    
        between_corr(sub,roi) = corr(squareform(control_RDMs(:,:,roi,sub))',squareform(challenge_RDMs(:,:,roi,sub))'); 
    end 
end 


%% load EEG RDMs 

eeg_path = '/Users/johannessinger/scratch/rcor_collab/derived/eeg';

eeg_fnames = dir(fullfile(eeg_path,'*.pkl')); 
eeg_fnames = {eeg_fnames.name}';

for i = 1:length(eeg_fnames)
    
    fid=py.open(char(fullfile(eeg_path,eeg_fnames(i))),'rb');
    data=py.pickle.load(fid);
end