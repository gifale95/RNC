% function decoding_sl(condition_names,labels,beta_dir,out_dir,cfg)
%
% Wrapper script for decoding using a searchlight.
%
% Input variables:
%   condition_names: Names of all regressors to be used for classification
%   labels: Labels that should be paired with condition_names (e.g. [-1 1])
%   beta_dir: name where results are which are used for classification
%   out_dir: name of folder where results are saved
%   cfg (optional): config variable used for the decoding

function results = decoding_nobetas(control_condition_names,challenge_condition_names,control_labels,challenge_labels,beta_avg_dir,out_dir,cfg,i_sub)

if ~exist('decoding_defaults.m','file')
    if ismac
        addpath('/Users/johannessinger/Documents/cloud_Berlin/Projekte/dfg/WP1/analysis_tools/tdt_3.999/decoding_toolbox');
    else
        addpath('/scratch/singej96/code_checking/reading/tdt_3.999/decoding_toolbox');
    end
end

if ~exist('cfg','var')
    cfg = decoding_defaults;
else
    cfg = decoding_defaults(cfg);
end

% Get subject number
try
    cfg.sn = str2double(regexp(beta_dir,'(?<=sub).[0-9]','match'));
catch
    warning('Could not extract subject number.')
end

% Set the output directory where data will be saved, e.g. 'c:\exp\results\buttonpress'
cfg.results.dir = out_dir;
cfg.results.overwrite = 1;

% Set the filepath where your SPM.mat and all related betas are, e.g. 'c:\exp\glm\model_button'
% done already

% Set the filename of your brain mask (or your ROI masks as cell matrix) 
% for searchlight or wholebrain e.g. 'c:\exp\glm\model_button\mask.img' OR 
% for ROI e.g. {'c:\exp\roi\roimaskleft.img', 'c:\exp\roi\roimaskright.img'}
try cfg.files.mask;
catch
    cfg.files.mask = fullfile(beta_dir,'mask.img');
    if ~exist(cfg.files.mask,'file')
        cfg.files.mask = fullfile(beta_dir,'mask.nii');
        if ~exist(cfg.files.mask,'file')
            error('Mask not found in %s',cfg.files.mask)
        end
    end
end

% Set additional parameters manually if you want (see decoding.m or
% decoding_defaults.m). Below some example parameters that you might want 
% to use:

% in case similarities should be calculated
if strcmpi(cfg.decoding.software,'similarity')
    cfg.decoding.method = 'classification';
end 

if  strcmpi(cfg.decoding.software, 'liblinear')
%use this for liblinear - might be faster for big datasets 
cfg.decoding.method = 'classification';
cfg.decoding.train.classification.model_parameters = '-s 1 -c 1 -q';
end 

% if cfg.noisenorm == 1 
%     
%     % These parameters carry out the multivariate noise normalization using the
% % residuals
% cfg.scale.method = 'cov'; % we scale by noise covariance
% cfg.scale.estimation = 'separate'; % we scale all data for each run separately while iterating across searchlight spheres
% cfg.scale.shrinkage = 'lw2'; % Ledoit-Wolf shrinkage retaining variances
% end 

% % scaling parameters if desired 
% cfg.scale.method = 'min0max1'; 
% cfg.scale.estimation = 'separate'; 

cfg.searchlight.unit = 'voxels'; %'mm'
cfg.searchlight.radius = 4; % 12; this will yield a searchlight radius of 12mm or 10mm 
cfg.searchlight.spherical = 1;
cfg.plot_design = 0;
cfg.verbose = 2; % you want all information to be printed on screen

% cfg.decoding.train.classification.model_parameters = '-s 0 -t 0 -c 1 -b 0 -q'; 
cfg.results.output = {'accuracy_pairwise_minus_chance'};

% Decide whether you want to see the searchlight/ROI/... during decoding
% cfg.plot_selected_voxels = 500; % 0: no plotting, 1: every step, 2: every second step, 100: every hundredth step...

% Add additional output measures if you like
% cfg.results.output = {'accuracy_minus_chance', 'AUC'}

%% Nothing needs to be changed below for a standard leave-one-run out cross
%% validation analysis.

% Set the following field:
% Full path to file names (1xn cell array) (e.g.
% {'c:\exp\glm\model_button\im1.nii', 'c:\exp\glm\model_button\im2.nii', ... }
for obj_i = 1:max(control_labels)
    for obj_j = obj_i+1:max(control_labels)
        
    control_beta_idxs_i = find(control_labels==obj_i); 
    control_beta_idxs_j = find(control_labels==obj_j);
    min_betas = min([length(control_beta_idxs_i),length(control_beta_idxs_j)]);
    control_sel_beta_idxs_i = datasample(control_beta_idxs_i,min_betas,'Replace',false); % 7 corresponds to the number of minimum images for one obj
    control_sel_beta_idxs_j = datasample(control_beta_idxs_j,min_betas,'Replace',false); % 7 corresponds to the number of minimum images for one obj
    challenge_beta_idxs_i = find(challenge_labels==obj_i); 
    challenge_beta_idxs_j = find(challenge_labels==obj_j);
    min_betas = min([length(challenge_beta_idxs_i),length(challenge_beta_idxs_j)]);
    challenge_sel_beta_idxs_i = datasample(challenge_beta_idxs_i,min_betas,'Replace',false); % 7 corresponds to the number of minimum images for one obj
    challenge_sel_beta_idxs_j = datasample(challenge_beta_idxs_j,min_betas,'Replace',false);
    ct = 1; 
    control_betas = [];
    challenge_betas = [];
    for i = 1:min_betas
    control_betas{ct} = fullfile(beta_avg_dir,[control_condition_names{control_sel_beta_idxs_i(i)},'_beta_avg.nii']);
    control_betas{ct+min_betas} = fullfile(beta_avg_dir,[control_condition_names{control_sel_beta_idxs_j(i)},'_beta_avg.nii']);
    challenge_betas{ct} = fullfile(beta_avg_dir,[challenge_condition_names{challenge_sel_beta_idxs_i(i)},'_beta_avg.nii']);
    challenge_betas{ct+min_betas} = fullfile(beta_avg_dir,[challenge_condition_names{challenge_sel_beta_idxs_j(i)},'_beta_avg.nii']);
    ct = ct +1;
    end 
    cfg.files.name = cat(1,control_betas(:),challenge_betas(:));
    cfg.files.name
    % assign labels and chunks 
    cfg.files.chunk = repmat([1:min_betas],1,4);
    cfg.files.label = repmat(repelem([1:2],min_betas),1,2);
    cfg.files.xclass = [ones(1,length(control_betas)), ones(1,length(challenge_betas))*2];
    cfg.files.twoway = 1; 

    % adjust out_dir 
    cfg.results.dir = fullfile(out_dir,[num2str(obj_i),'_vs_',num2str(obj_j)]);

    % This creates the leave-one-run-out cross validation design:
    cfg.design = make_design_xclass_cv(cfg); 
    
    % start decoding
    results = decoding(cfg);
    end 
end 

% and the other two fields if you use a make_design function (e.g. make_design_cv)
%
% (1) a nx1 vector to indicate what data you want to keep together for 
% cross-validation (typically runs, so enter run numbers)
% for i = 0:max(labels) 
%     cnt = sum(labels==i);     
%     cfg.files.chunk = [cfg.files.chunk, [1:cnt]]; %repmat([1:avg_size],1,242)';
% end
%
% (2) any numbers as class labels, normally we use 1 and -1. Each file gets a
% label number (i.e. a nx1 vector)

% split up the decoding in all pairwise comparisons 


% %cfg.design.unbalanced_data = 'ok';  
% 
% if cfg.perm == 1
%  
%     for i = 1:50
%         cfg.design.label(:,i) = cfg.design.label(:,1);
%         cfg.design.set(i) = 1; 
%         train = zeros(1,avg_size);
%         train(randi(avg_size)) = 1; 
%         test = (train ==0); 
%         cfg.design.train(:,i) = repmat(train,1,60); 
%         cfg.design.test(:,i) = repmat(test,1,60); 
%     end
% end 
% 
% if ismac | cfg.parallel ==0
% 
%     % Run decoding
% 
%     results = decoding(cfg);
%  
% 
% elseif isunix && cfg.parallel ==1
%     
%     % get the number of searchlights for parallelization 
%     mask = spm_read_vols(spm_vol(cfg.files.mask{1}));
%     num_searchlights = size(mask(mask==1),1);
%     % start the parallel pool
%     pool =  parcluster('local');
%     % get number of workers (cpus)
%     num_workers = pool.NumWorkers;
%     searchlights_per_worker = ceil(num_searchlights/num_workers); % Divide the task up into the number of workers
%     fprintf('Starting searchlight decoding on %i parallel cores with %i searchlights for each worker', num_workers, searchlights_per_worker)
%     parfor crun = 1:num_workers
%     results{crun} = decoding_parallel_wrapper(cfg,searchlights_per_worker,crun)
%     end
%     all_results = results{1};
%     for crun = 2:num_workers
%     all_results.decoding_subindex = [all_results.decoding_subindex; results{crun}.decoding_subindex];
%     all_results.accuracy_pairwise_minus_chance.output(results{crun}.decoding_subindex) = results{crun}.accuracy_pairwise_minus_chance.output(results{crun}.decoding_subindex); 
%     %all_results.other_average.output(results{crun}.decoding_subindex) = results{crun}.other_average.output(results{crun}.decoding_subindex);
%     end
%     results = all_results;
%     disp('Decoding on the whole brain complete, saving results')
%     sub_config = config_subjects_visdecmak(); 
%     combine_write_decoding_results(sub_config, i_sub,results, fullfile(cfg.results.dir,'res_accuracy_pairwise_minus_chance.nii'))
%     save(fullfile(cfg.results.dir,'res_accuracy_pairwise_minus_chance.mat'),'results')
%     %assert(sum(cellfun(@isempty,all_results.oth.output))==0,‘Results Output not completely filled despite completion of the parallel loop - please check’)
%     delete(fullfile(cfg.results.dir,'parallel_loop*.mat'))
%     delete(fullfile(cfg.results.dir,'parallel_loop*.nii'))
% 
% end      
end 