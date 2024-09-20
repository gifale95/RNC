%% Script for checking the movement parameter deviations from the first volume in a block for all subjects

% clear all 
% clc

% add all the helper functions 
addpath(genpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri'))

% specify path for the decoding toolbox here (which is needed for later
% steps in the analysis) 
addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'))

% path for tapas toolbox - for denoising 
addpath(genpath('/data/pt_02348/objdraw/fmri/tapas-master'))

% get config for experiment 

cfg = config_subjects_objdraw();

% intiliaze all deviation paramaters 
all_dev_params = [];

% run loop over subject 

for i_sub = 1:length(cfg.subject_indices)

sub_dir = fullfile(cfg.dirs.derived_dir,sprintf('sub%02d',i_sub));

fname = fullfile(sub_dir,'alldata','parameters',sprintf('rp_sub%02d.mat',i_sub));

try
% load movement parameters 
load(fname)
catch 
    continue 
end 

% now check the deviation for every run separately 
dev_params = [];

for run = 1:length(ct)
    if run == 1
        for vol = 2:ct(run)
            
            dev_params = cat(1,dev_params,mparams(2,:) -mparams(vol-1,:));
        end
    elseif run < length(ct)
        for vol = ct(run-1)+2:ct(run)
            dev_params = cat(1,dev_params, mparams(ct(run-1),:) -mparams(vol-1,:));
        end
    else
        for vol = ct(run-1)+2:ct(run)
            dev_params = cat(1,dev_params, mparams(ct(run-1),:) -mparams(vol-1,:));
        end
        for vol = ct(run)+2:ct(run)+250
            dev_params = cat(1,dev_params, mparams(ct(run),:) -mparams(vol-1,:));
        end
    end
end

all_dev_params = cat(3, all_dev_params, dev_params);
% plot deviations 
figure
plot(dev_params(:,1:3))
ylim([-2 2])
legend('x translation','y translation','z translation');
title(['Deviations form first volume in block for subject ', cfg.sub(i_sub).pid])

if any(any(abs(dev_params(:,1:3)) > 2))
    fprintf('Subject %s should be excluded from the analysis\n', cfg.sub(i_sub).pid)
end
end 