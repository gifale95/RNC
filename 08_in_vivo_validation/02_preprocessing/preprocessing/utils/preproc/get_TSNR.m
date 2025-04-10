%% function to create TSNR maps and median absolute difference values for quality check 

function tsnr_mean = get_TSNR(cfg,i_sub,prefix,include_loc,maskname) 

% Input variables:
%   cfg (passed)
%   i_sub: Subject number
%   prefix: Prefix of files to check (e.g. 'f')
%   include_loc: if localizer data should be included 
%   plot_on: if 0, don't plot
%   maskname: (optional): path to mask; don't take into account out of mask regions

%% Get volumes

out_im = []; % init
out_slice = []; % init

sub_dir = fullfile(cfg.dirs.derived_dir,sprintf('sub%02d',i_sub));
prefix = [prefix cfg.prefix];


P = []; % all headers
ct = []; % counter of where a run ends

if include_loc
    % Localizer runs
        fname_path = fullfile(sub_dir,'alldata','localizer');
        fname = spm_select('fplist',fname_path,['^' prefix '.*\.nii$']);
        P = [P; spm_vol(fname)]; % get headers
        ct = [ct length(P)];
end

% Functional runs
for i_run = cfg.sub(i_sub).run_indices
    fname_path = fullfile(sub_dir,'alldata',sprintf('run%02d',i_run));
    fname = spm_select('fplist',fname_path,['^' prefix '.*\.nii$']);
    P = [P; spm_vol(fname)]; % get headers
    ct = [ct length(P)];
end

n_runs = length(ct);
ct = ct(1:end-1); % remove last entry because we don't need a line at the end
run_ind = [1 ct+1; ct length(P)]; % where each run starts and ends

% if there is a mask load it 
if exist('maskname','var')
    maskvol = spm_read_vols(spm_vol(maskname));
else
    maskvol = ones(P(1).dim);
end

maskvol = logical(maskvol);

% loop through runs 
for i_run = 1:n_runs
    
    hdrs = P(run_ind(1,i_run):run_ind(2,i_run));
    n_vols = size(hdrs,1);
    if isempty(n_vols) || n_vols < 2, return, end
    n_vols = n_vols-1;

    % initialize vols 
    vols = NaN([cfg.dim, n_vols]); 
    fprintf('Loading vols from %i to %i\n', run_ind(1,i_run),run_ind(2,i_run)); 

    vol_ct = 1; 
for i = run_ind(1,i_run):run_ind(2,i_run)
    
    this_vol = spm_read_vols(P(i)); 
    vols(:,:,:,vol_ct) = this_vol.*maskvol; 
    vol_ct = vol_ct+1; 
end 

% compute tsnr 
fprintf('Computing TSNR for run %i\n', i_run); 
[tsnr,tsnr_mean(i_run)] = computetemporalsnr_adapted(vols);
fprintf('Median TSNR for run %i is %2f\n', i_run,tsnr_mean(i_run)); 


% save tsnr map 
tsnr_hdr = P(1); 
tsnr_hdr.fname = fullfile(sub_dir,'alldata', 'other',sprintf('tsnr_map_run_%02i.nii',i_run)); 
spm_write_vol(tsnr_hdr,tsnr); 
end 
end 


