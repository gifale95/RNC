% function files = select_files(cfg,i_sub,prefix,include_loc,concatenate_runs)
% 
% Get file list in the format that is good for matlabbatch
% Inputs:
%   cfg: contains the runs that should be included for each subject, the
%   experiment-specific prefix that is contained in all file names and the 
%   directory of the data (e.g.
%           cfg.prefix = 'myexp';
%           cfg.suffix = 'img';
%           cfg.dirs.data_dir = 'c:\myexperiment';
%           cfg.sub(1).run_indices = [1 3 4 5];)
%   i_sub: Index of current subject (e.g. 12 for subject 12)
%   prefix: gives the prefix of the functional data (e.g. f, rf, etc.)
%   Input include_loc includes localizer run or not
%   Input concatenate_runs puts runs together or not (e.g. for normalization or
%       smoothing)
%
% If you never use a localizer run, you could consider removing the
% variable include_loc from the function and deleting the part starting
% with "if include_loc".

function files = select_files_adapted(cfg,i_sub,prefix,include_loc,concatenate_runs)

files = {};
prefix = [prefix cfg.prefix];
prefix_localizer = [prefix cfg.prefix_localizer];
suffix = cfg.suffix;
subind = sprintf('sub%02d',i_sub);
sub_dir = fullfile(cfg.dirs.derived_dir,subind);


if include_loc(1) == 1
    
    % Object localizer
    
    if cfg.sub(i_sub).import.localizer
        
        run_dir = fullfile(sub_dir,'alldata','localizer');
        tmp = spm_select('FPList',run_dir,['^' prefix_localizer '.*\.' suffix '$']);
        
        if ~isempty(tmp)
            
            tmp2 = cell(size(tmp,1),1);
            for i = 1:size(tmp,1)
                tmp2{i} = [tmp(i,:) ',1'];
            end
            
            files(end+1) = {tmp2};
            
        end
    end
end 


% pRF runs
n_scans = cfg.n_scans_prf_experiment;
for i_run = 1:cfg.sub(i_sub).import.prf_experiment_runs
    runind = sprintf('run%02d',i_run);
    run_dir = fullfile(sub_dir,'alldata',runind);
    tmp = spm_select('FPList',run_dir,['^' prefix '.*\.' suffix '$']);
    tmp2 = cell(n_scans,1);
    for i = 1:n_scans
        tmp2{i} = [tmp(i,:)];
	end
    files(end+1) = {tmp2}; %#ok<AGROW>
end

% UC runs
n_scans = cfg.n_scans_uc_experiment;
for i_run = 4:3+cfg.sub(i_sub).import.uc_experiment_runs
    runind = sprintf('run%02d',i_run);
    run_dir = fullfile(sub_dir,'alldata',runind);
    tmp = spm_select('FPList',run_dir,['^' prefix '.*\.' suffix '$']);
    tmp2 = cell(n_scans,1);
    for i = 1:n_scans
        tmp2{i} = [tmp(i,:)];
	end
    files(end+1) = {tmp2}; %#ok<AGROW>
end


% MC runs
n_scans = cfg.n_scans_mc_experiment;
for i_run = 14:13+cfg.sub(i_sub).import.mc_experiment_runs
    runind = sprintf('run%02d',i_run);
    run_dir = fullfile(sub_dir,'alldata',runind);
    tmp = spm_select('FPList',run_dir,['^' prefix '.*\.' suffix '$']);
    tmp2 = cell(n_scans,1);
    for i = 1:n_scans
        tmp2{i} = [tmp(i,:)];
	end
    files(end+1) = {tmp2}; %#ok<AGROW>
end


    
    % Object localizer - second session
    
	if include_loc(2) == 1
		
    if cfg.sub(i_sub).import.second_localizer
        
        run_dir = fullfile(sub_dir,'alldata','localizer_second');
        tmp = spm_select('FPList',run_dir,['^' prefix_localizer '.*\.' suffix '$'])
        
        if ~isempty(tmp)
            
            tmp2 = cell(size(tmp,1),1);
            for i = 1:size(tmp,1)
                tmp2{i} = [tmp(i,:) ',1'];
            end
            
            files(end+1) = {tmp2};
            
        end
	end
	end


if exist('concatenate_runs','var') && concatenate_runs == 1
    newfiles = [];
    for i = 1:length(files)
        newfiles = [newfiles; files{i}]; %#ok<AGROW>
    end
    files = newfiles;
end
files