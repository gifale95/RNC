function results = decoding_parallel_wrapper(cfg,searchlights_per_worker,worker_number,misc)
% in case no residuals are given set misc to empty variable
if nargin < 4
    misc = []; 
end 
cfg.searchlight.subset = ((worker_number-1)*searchlights_per_worker)+1:worker_number*searchlights_per_worker;
cfg.results.resultsname = cellstr(['parallel_loop_' num2str(worker_number)]);
spm_dir = '/scratch/pablooyarzo/proj1/fMRI/analysis_tools/spm12';%fileparts(which('spm'));
addpath(spm_dir)     
 % Your SPM path for the workers
spm('ver'); % Needed or sometimes the decoding toolbox complains in parallel that SPM is not initialised.
if cfg.noisenorm == 0
    results = decoding(cfg);
elseif cfg.noisenorm ==1
    results = decoding(cfg,[],misc); 
end 
end 