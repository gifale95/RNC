function matlabbatch = firstlevel_localizer(cfg,prefix,results_dir,onsname,i_sub,mparams,physio,glmdenoise,n_slices)

% function matlabbatch = firstlevel(cfg,prefix,results_dir,onsname,i_sub,mparams,n_slices)

% cfg: passed from config_subjects
% prefix: prefix of file (e.g. 'arf')
% results_dir: Subdirectory where results are written (e.g. 'myresult')
% onsname: name of onsets (e.g. 'buttonpresses')
% i_sub: subject number
% mparams: should motion parameters be included (1 or 0)
% physio: should physiological parameters be included (1 or 0)
% glmdenoise: name of noise regressors or 0
% n_slices: number of slices (optional, if not provided will be extracted from file)

mparams; % check if exists
physio;
glmdenoise;

if sum(mparams+physio)>1 || (sum(mparams+physio)==1 && sum(glmdenoise) ~= 0)
    error('At the moment, only one of mparams, physio and glmdenoise can be chosen!')
end

sub_dir = cfg.sub(i_sub).dir;
ons_dir = fullfile(sub_dir,'onsets',onsname);

if ~isdir(results_dir), mkdir(results_dir), end

matlabbatch{1}.spm.stats.fmri_spec.dir = {results_dir};
matlabbatch{1}.spm.stats.fmri_spec.timing.units = 'secs';
matlabbatch{1}.spm.stats.fmri_spec.timing.RT = cfg.TR;

ct = 1;

% check if second localizer is used or not 
loc_count = length(cfg.sub(i_sub).import.localizer) + length(cfg.sub(i_sub).import.second_localizer);

for loc_idx = loc_count

if loc_idx ==1
    % get file names
    run_dir = fullfile(sub_dir,'alldata','localizer');
    tmp = spm_select('FPList',run_dir,['^' prefix '.*\.(img|nii)$']);
elseif loc_idx==2
    % get file names
    run_dir = fullfile(sub_dir,'alldata','localizer_second');
    tmp = spm_select('FPList',run_dir,['^' prefix '.*\.(img|nii)$']);
end 
if isempty(tmp)
    error('No files found with prefix %s in %s',prefix,run_dir)
end
files = cell(size(tmp,1),1);
for i = 1:size(tmp,1)
    files{i} = [tmp(i,:) ',1'];
end

matlabbatch{1}.spm.stats.fmri_spec.sess(ct).scans = files;

if ~exist('n_slices','var')
    hdr = spm_vol(files{1});
    n_slices = hdr.dim(3);
end
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t = n_slices;
matlabbatch{1}.spm.stats.fmri_spec.timing.fmri_t0 = ceil(n_slices/2);

matlabbatch{1}.spm.stats.fmri_spec.sess(ct).cond = struct('name', {}, 'onset', {}, 'duration', {}, 'tmod', {}, 'pmod', {}, 'orth', {});
if loc_idx ==1 
    matlabbatch{1}.spm.stats.fmri_spec.sess(ct).multi = {fullfile(ons_dir,sprintf('onsets_%s.mat',onsname))};
elseif loc_idx ==2 
    matlabbatch{1}.spm.stats.fmri_spec.sess(ct).multi = {fullfile(ons_dir,sprintf('onsets_%s_second.mat',onsname))};
end
matlabbatch{1}.spm.stats.fmri_spec.sess(ct).regress = struct('name', {}, 'val', {});
if mparams
    rp_name = spm_select('fplist',run_dir,'^rp_.*\.txt$');
    matlabbatch{1}.spm.stats.fmri_spec.sess(ct).multi_reg = {rp_name};
elseif physio
    physio_path = fullfile(cfg.sub(i_sub).dir,'alldata','parameters');
    physio_name = fullfile(physio_path,sprintf('physioreg_run%02i.mat',i_run));
    if ~exist(physio_name,'file')
        error('No physio-logging found for subject %i and run %i',i_sub,i_run)
        physio_name = '';
    end
    matlabbatch{1}.spm.stats.fmri_spec.sess(ct).multi_reg = {physio_name};
elseif glmdenoise
    glmdenoise_path = fullfile(cfg.sub(i_sub).dir,'alldata','parameters');
    glmdenoise_name = fullfile(glmdenoise_path,glmdenoise,sprintf('noisereg_run%02i.mat',i_run));
    if ~exist(glmdenoise_name,'file')
        glmdenoise_name = fullfile(glmdenoise_path,sprintf('glmdenoise_%s_run%02i.mat',glmdenoise,i_run));
    end
    matlabbatch{1}.spm.stats.fmri_spec.sess(ct).multi_reg = {glmdenoise_name};
else
    matlabbatch{1}.spm.stats.fmri_spec.sess(ct).multi_reg = {''};
end
matlabbatch{1}.spm.stats.fmri_spec.sess(ct).hpf = 128;
ct = ct+1;
end

matlabbatch{1}.spm.stats.fmri_spec.fact = struct('name', {}, 'levels', {});
matlabbatch{1}.spm.stats.fmri_spec.bases.hrf.derivs = [0 0];
matlabbatch{1}.spm.stats.fmri_spec.volt = 1;
matlabbatch{1}.spm.stats.fmri_spec.global = 'None';
matlabbatch{1}.spm.stats.fmri_spec.mthresh = -Inf;
matlabbatch{1}.spm.stats.fmri_spec.mask = {fullfile(cfg.sub(i_sub).dir,'roi','brainmask.nii')};
matlabbatch{1}.spm.stats.fmri_spec.cvi = 'none'; % alternative: 'AR(1)'; or 'none';

matlabbatch{2}.spm.stats.fmri_est.spmmat(1) = cfg_dep;
matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tname = 'Select SPM.mat';
matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tgt_spec{1}(1).name = 'filter';
matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tgt_spec{1}(1).value = 'mat';
matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tgt_spec{1}(2).name = 'strtype';
matlabbatch{2}.spm.stats.fmri_est.spmmat(1).tgt_spec{1}(2).value = 'e';
matlabbatch{2}.spm.stats.fmri_est.spmmat(1).sname = 'fMRI model specification: SPM.mat File';
matlabbatch{2}.spm.stats.fmri_est.spmmat(1).src_exbranch = substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1});
matlabbatch{2}.spm.stats.fmri_est.spmmat(1).src_output = substruct('.','spmmat');
matlabbatch{2}.spm.stats.fmri_est.method.Classical = 1;
end