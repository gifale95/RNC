% function cfg = analysis(i_sub,cfg)
%
% This function runs the preprocessing and subject-level statistics
% of one subject i_sub. If i_sub is missing, the planned analyses are
% printed, but nothing is executed.
% A separate function (analysis2) executes group-level analyses
% All settings are made at the corresponding part of the function or in
% config_subjects_objdraw.m
% You need to set the spm path for the function to work ("edit analysis_visdecmak_fmri.m")

function cfg = fmri_preprocessing(i_sub,cfg)

% set this variable to load SPM toolboxes only once per loop
%global loadtoolbox

%% set preprocessing

cfg.do.import_nii = 1; % for import of .nii images (if dcm2niix is used)
cfg.do.import_nii_second = 1; % for import of .nii images from the second session
cfg.do.realign_init = 1; % initialization of the realignment
cfg.do.plot_realign = 1; % plot realignment parameters
cfg.do.eyemask_test = 1; % visually check the eyemasking of the mean image
cfg.do.eyemask = 1; % create eyemask if check was ok
cfg.do.datadiagnostics = 1; % compute the data-diagnostics for one dataset
cfg.do.plot_datadiagnostics = 1; % only for plotting and reporting outlier slices
cfg.do.replace_volumes = 0; % replace entire volumes with high variance %%%%TODO: SOMETHING IS WRONG HERE - CHECK THAT!!%%%
cfg.do.realign = 1; % actually do the realignment
cfg.do.slicetiming = 1; % slicetiming correction
cfg.do.create_fieldmap = 1; % prepare fieldmap
cfg.do.apply_fieldmap = 1; % apply fieldmap
cfg.do.apply_fieldmap_mean = 1; % apply fieldmap to mean image
cfg.do.coreg = 1; % coregister to unwarped mean!
cfg.do.nii_3Dto4D = 1;
cfg.do.nii_mean = 1;
cfg.do.segment = 0; % segment the structural image
cfg.do.check_segment = 0; % visually check segmentation
cfg.do.smoothing =0; % smooth the functional images
cfg.do.normalize = 0; %normalize functional images
cfg.do.tapas_denoising = 0; %acompcor denoising
cfg.do.glmdenoise = 0; % GLM denoise(Kay, 2013)
cfg.do.glmsingle = 0; %GLM single by Kendrick Kay

%% set firstlevel

cfg.do.firstlevel_overwrite = 0; % if existing analyses should be overwritten
cfg.do.contrasts_only = 0; % if this is ticked, the firstlevel will not be executed again

cfg.do.firstlevel_localizer = 0; % compute first level GLM for the localizer run

cfg.do.firstlevel_no_hrf_fitting = 0; % compute first level GLM for the experimental blocks without voxel-wise HRF fitting

cfg.do.firstlevel_hrf_fitting = 0; % compute first level GLM for the experimental blocks with voxel-wise HRF fitting

cfg.do.eval_hrf_fitting = 0; % evaluate the GLMs after computing them with different HRFs

cfg.do.only_eval_hrf_fitting = 0; % only evaluate the GLMs after computing them with different HRFs

cfg.do.first_level_contrasts = 0; % compute first level contrasts

cfg.do.normalize_first_level = 0; % normalize the betas computed in the first level GLM

%% ROI definition

cfg.do.write_ROI_EVC = 0; % write EVC ROI mask (based on Glasser Atlas and Image Contrast)
cfg.do.write_ROI_LOC = 0; % write LOC mask
cfg.do.write_ROI_PPA = 0; % PPA mask based on Scene vs all Contrast and kanwisher mask
cfg.do.check_ROI_overlap = 0; % check for overlap between ROIs and remove overlapping voxels
cfg.do.write_brain_mask = 0; % take an explicit brain mask in the MNI space and transform into individual space

%% set decoding

cfg.do.unn = 0; % apply univariate noise normalization and save betas
cfg.do.decoding_obj = 0; % object decoding for challenge and control images separately
cfg.do.avg_betas = 0;         % average betas for higher SNR for decoding
cfg.do.decoding_obj_nobetas =0; % do manmade/natural decoding with betas not from SPM
cfg.do.crossdecoding_obj_nobetas =0; % do object cross-decoding with betas not from SPM
cfg.do.decoding_nobetas_normalized = 0; % do manmade/natural decoding with normalized betas
cfg.do.decoding_splithalf = 0; % do manmade/natural decoding with data averaged in two halfs
cfg.do.decoding_nocrossval = 0; % do manmade/natural decoding with all trials for training and one mean trial for testing
cfg.do.decoding_nobetas_controversial = 0; % crossdecoding with controversial stimuli
cfg.do.decoding_nobetas_controversial_congruent = 0; % crossdecoding with only congruent controversial stimuli
cfg.do.decoding_nobetas_controversial_congruent_manmade = 0; % crossdecoding only with congruent manmae-manmade stimuli
cfg.do.decoding_nobetas_only_controversial =0; % crossdecoding with only congruent controversial stimuli

%% set RSA

cfg.do.decoding_similarity_crossnobis = 0; %compute crossvalidated mahalonobis distances
cfg.do.decoding_similarity_pearson = 0; %compute correlation distances after noise normalization
cfg.do.decoding_similarity_all = 0; % compute similarity matrices with betas from all conditions together (BIG RDMs)

%% normalize (and smooth) firstlevel and decodings

cfg.do.smooth_norm_res=0; % normalize + smooth first level results
cfg.do.subtract_chance = 0 ; % specify if chance level should be subtracted from the smoothed and normalized results -> for searchlight results
cfg.do.norm_mask_for_GSS = 0; % normalize first level contrasts and localizer contrasts and create a localizer mask for the GSS method for defining fROIs

%% if no subject is provided as input, return list of analyses to conduct

if ~exist('i_sub','var')
    fn = fieldnames(cfg.do);
    for i_field = 1:size(fn,1)
        if cfg.do.(fn{i_field})
            if ~exist('str','var')
                str = ['cfg.do.' fn{i_field}];
            else
                str = char(str, ['cfg.do.' fn{i_field}]);
            end
        end
    end
    disp('Analyses to conduct:')
    if ~exist('str','var'), disp('none.')
    else disp(str)
    end
    return
end

%% Create cfg for all subjects

if exist('cfg','var')
    cfg = config_fmri_preprocessing(cfg);
else
    cfg = config_fmri_preprocessing;
end

fpath = mfilename('fullpath');
[fpath fname fext] = fileparts(fpath); %#ok
cfg.dirs.function_dir = fpath;

clear cfg_item % remove e.g. when spm12 was loaded in advance

addpath(cfg.dirs.spm_dir); %#ok<*MCAP>
addpath(genpath(cfg.dirs.code_dir))
%addpath(genpath(cfg.dirs.function_dir)); % if you have some stats scripts

data_dir = cfg.dirs.data_dir;

%% Set paths

if ~exist(data_dir,'dir'), mkdir(data_dir), end



%% ==========================================================
%% PREPROCESSING
%% ==========================================================

%% import niftis (in case niftis are created with dcm2niix)

if cfg.do.import_nii
    
    %plot_sn(i_sub);
    fprintf('Importing Nifti and Json files...\n')
    
    fprintf('\nRunning Subject %i\n\n',i_sub)
    input_dir = fullfile(data_dir,cfg.sub(i_sub).id,'ses-01'); % Setting folder where the .nii files are located
    disp('Getting Nifti and Json filenames...')
	% Anatomical
    nii_anat = dir(fullfile(input_dir, 'anat', '*.nii')); % Selecting .nii files
    nii_anat ={nii_anat.name}';
    json_anat = dir(fullfile(input_dir, 'anat', '*.json'));
    json_anat ={json_anat.name}';
	% Field map
    nii_fmap = dir(fullfile(input_dir, 'fmap', '*.nii')); % Selecting .nii files
    nii_fmap ={nii_fmap.name}';
    json_fmap = dir(fullfile(input_dir, 'fmap', '*.json'));
    json_fmap ={json_fmap.name}';
	% Functional
    nii_func = dir(fullfile(input_dir, 'func', '*.nii')); % Selecting .nii files
    nii_func ={nii_func.name}';
    json_func = dir(fullfile(input_dir, 'func', '*.json'));
    json_func ={json_func.name}';
    
    disp('Converting and saving the nii and json files...')
    convert_nii_to_spm_visdecmak(nii_anat,json_anat, nii_fmap, json_fmap, ...
		nii_func, json_func, input_dir, cfg, i_sub);
    
    clear hdr; % free memory
    
    disp('done')
    
end

if cfg.do.import_nii_second
    
    %plot_sn(i_sub);
    fprintf('Importing Nifti and Json files for second session...\n')
    
    fprintf('\nRunning Subject %i\n\n',i_sub)
    input_dir = fullfile(data_dir,cfg.sub(i_sub).id,'ses-02'); % Setting folder where the .nii files are located
    disp('Getting Nifti and Json filenames...')
	% Anatomical
    nii_anat = dir(fullfile(input_dir, 'anat', '*.nii')); % Selecting .nii files
    nii_anat ={nii_anat.name}';
    json_anat = dir(fullfile(input_dir, 'anat', '*.json'));
    json_anat ={json_anat.name}';
	% Field map
    nii_fmap = dir(fullfile(input_dir, 'fmap', '*.nii')); % Selecting .nii files
    nii_fmap ={nii_fmap.name}';
    json_fmap = dir(fullfile(input_dir, 'fmap', '*.json'));
    json_fmap ={json_fmap.name}';
	% Functional
    nii_func = dir(fullfile(input_dir, 'func', '*.nii')); % Selecting .nii files
    nii_func ={nii_func.name}';
    json_func = dir(fullfile(input_dir, 'func', '*.json'));
    json_func ={json_func.name}';
    
    disp('Converting and saving the nii and json files...')
    [hdr,cfg] = convert_nii_to_spm_visdecmak_second(nii_anat,json_anat, ...
		nii_fmap, json_fmap, nii_func, json_func, input_dir, cfg, i_sub);
    
    clear hdr; % free memory
    
    disp('done')
    
end



%% Initial realignment
% Run realignment the quick and dirty way to get outlier volumes (= volumes
% where movement ocurred), write mean image only

if cfg.do.realign_init
    
    %plot_sn(i_sub);
    include_loc = [0 0]; % two entries for first and second session - 1 indicates include, 0 dont include
    
    % Initialize the jobmanager
    spm('defaults','fmri')
    spm_jobman('initcfg');
    
    clear matlabbatch
    
    matlabbatch{1}.spm.spatial.realign.estwrite.data = select_files_adapted(cfg,i_sub,'f',include_loc);
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.9;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 4;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 0 ;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 4;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = {''};
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [0 1];
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 0 ;
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';
    
    spm_jobman('run',matlabbatch)
    
    %copy mean file to mean folder
    if include_loc
        movefile(fullfile(cfg.sub(i_sub).dir,'alldata','localizer','*mean*'),fullfile(cfg.sub(i_sub).dir,'alldata','mean'));
    else
        movefile(fullfile(cfg.sub(i_sub).dir,'alldata','run01','*mean*'),fullfile(cfg.sub(i_sub).dir,'alldata','mean'));
    end
end

%% Create image of realignment parameters

if cfg.do.plot_realign
    
    %plot_sn(i_sub);
    prefix = 'f';
    include_loc = 0;
    plot_on = 1 ;
    plot_rp(cfg,i_sub,prefix,include_loc,plot_on)
    print(fullfile(cfg.sub(i_sub).dir,'alldata','other', ['realign_init.jpg']), ...
        '-djpeg', '-r300')
end

%% Check if eyes are found

if cfg.do.eyemask_test
    
    %plot_sn(i_sub);
    prefix = 'f';
    
    k = cfg.sub(i_sub).preproc.eyesize_range;
    mean_dir = fullfile(cfg.sub(i_sub).dir,'alldata','mean');
    maskname = spm_select('fplist',mean_dir,['^mean' prefix cfg.prefix '.*\.nii$']);
    remove_eyes(k(1),k(2),k(3),maskname,0,1) ;  % for testmode, set last digit to 1
    
    % save the eyemask image
    print(fullfile(cfg.sub(i_sub).dir,'alldata','other', ['eyemask_test.jpg']), ...
        '-djpeg', '-r300')
end

%% Create weight maps with eyes removed

if cfg.do.eyemask
    
    %plot_sn(i_sub);
    prefix = 'f';
    
    k = cfg.sub(i_sub).preproc.eyesize_range;
    mean_dir = fullfile(cfg.sub(i_sub).dir,'alldata','mean');
    maskname = spm_select('fplist',mean_dir,['^mean' prefix cfg.prefix '.*\.nii$']);
    remove_eyes(k(1),k(2),k(3),maskname,0,0) ;  % for testmode, set last digit to 1
    
end

%% Run data diagnostics

if cfg.do.datadiagnostics
    
    %plot_sn(i_sub);
    prefix = 'f';
    maskprefix = ['emean' prefix cfg.prefix];
    % if maskname is not included, the eyes - which cause large amounts of
    % variance - are still in, obscuring outlier volumes
    maskname = spm_select('fplist',fullfile(cfg.sub(i_sub).dir,'alldata','mean'),['^' maskprefix '.*\.nii$']);
    check_data = 1;
    include_loc = 0 ;
    plot_on = 0 ;
    check_slices(cfg,i_sub,prefix,include_loc,check_data,plot_on,maskname);
    
end

if cfg.do.plot_datadiagnostics
    
    %plot_sn(i_sub);
    prefix = 'f';
    maskprefix = ['emean' prefix cfg.prefix];
    % if maskname is not included, the eyes - which cause large amounts of
    % variance - are still in, obscuring outlier volumes
    maskname = spm_select('fplist',fullfile(cfg.sub(i_sub).dir,'alldata','mean'),['^' maskprefix '.*\.nii$']);
    check_data =  1;
    include_loc = 0;
    plot_on = 1 ;
    [outlier_im, outlier_slices] = check_slices(cfg,i_sub,prefix,include_loc,check_data,plot_on,maskname);
    
    fprintf('Percentage outlier images of subject %i:\n',i_sub)
    if isempty(outlier_im)
        disp('none')
    else
        disp('Outlier images:')
        disp(outlier_im)
        %disp(num2str(length(outlier_im)/(cfg.n_scans_experiment*10)*100))
    end
    %     fprintf('Outlier slices of subject %i:\n',i_sub)
    %     if isempty(outlier_slices)
    %         disp('none')
    %     else
    %         str1 = floor(outlier_slices);
    %         str2 = 100*(outlier_slices-floor(outlier_slices));
    %         for i_str = 1:length(outlier_slices)
    %             disp([num2str(str1(i_str)) ':' num2str(str2(i_str))])
    %         end
    %     end
    %     disp('Please check if outlier images can be reduced to outlier slice (e.g. through spike in a slice only)!')
    
    print(fullfile(cfg.sub(i_sub).dir,'alldata','other', ['data_diagnostics.jpg']), ...
        '-djpeg', '-r300')
    
    
end

%% Remove bad volumes (copy them to separate folder, then replace them by surrounding volumes [bicubic spline])

if cfg.do.replace_volumes
    
    %plot_sn(i_sub);
    prefix = 'f';
    include_loc=0;
    
    replace_volume(cfg,i_sub,prefix,include_loc);
    
end

%% Realignment

if cfg.do.realign
    
    %plot_sn(i_sub);
    prefix = 'f';
    include_loc = [0 0] ; % should a localizer run (or several) be included in preprocessing?
    
    % Initialize the jobmanager
    spm('defaults','fmri')
    %initjobman; % see function at end of script (to initialize only once)
    
    clear matlabbatch
    
    matlabbatch{1}.spm.spatial.realign.estwrite.data = select_files_adapted(cfg,i_sub,prefix,include_loc);
    matlabbatch{1}.spm.spatial.realign.estwrite.data
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.quality = 0.95;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.sep = 2;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.fwhm = 5;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.rtm = 0 ; % no realignment to mean - 1 means realignment to mean, 0 means realignment to first image of session
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.interp = 2;
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.wrap = [0 0 0];
    mean_dir = fullfile(cfg.sub(i_sub).dir,'alldata','mean');
    fname = spm_select('fplist',mean_dir,['^emean' prefix cfg.prefix '.*.nii$']);
    matlabbatch{1}.spm.spatial.realign.estwrite.eoptions.weight = {[fname ',1']};
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.which = [2 1]; % reslice all plus mean
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.interp = 4;
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.mask = 1 ;
    matlabbatch{1}.spm.spatial.realign.estwrite.roptions.prefix = 'r';
    
    spm_jobman('run',matlabbatch)
    
end

%% Slice timing correction

if cfg.do.slicetiming
    
    %plot_sn(i_sub);
    include_loc = [0 0] ;
    spm('defaults','fmri')
    %initjobman; % Initialize the jobmanager
    
    prefix = 'rf'; % 'rf'
    cfg.suffix = '(nii|img)';
    
    n_slices = cfg.n_slices;
    TR = cfg.TR;
    
    clear matlabbatch
    
    matlabbatch{1}.spm.temporal.st.scans = select_files_adapted(cfg,i_sub,prefix,include_loc);
    matlabbatch{1}.spm.temporal.st.nslices = n_slices;
    matlabbatch{1}.spm.temporal.st.tr = TR;
    matlabbatch{1}.spm.temporal.st.ta = TR - (TR/n_slices);
    
    switch lower(cfg.sliceorder)
        case 'ascending'
            so = 1:n_slices;
        case 'descending'
            so = n_slices:-1:1 ;
        case 'interleaved'
            if mod(cfg.n_slices,2)
                so = [1:2:n_slices 2:2:n_slices];
            else
                warning('Using settings for Siemens, if no Siemens Scanner is used please find this message and change the script!') %#ok<WNTAG>
                so = [2:2:n_slices 1:2:n_slices];
            end
        case 'interleaved descending'
            if mod(cfg.n_slices,2)
                so = [n_slices:-2:1 n_slices-1:-2:1];
            else
                warning('Using settings for Siemens, if no Siemens Scanner is used please find this message and change the script!') %#ok<WNTAG>
                so = [n_slices-1:-2:1 n_slices:-2:1];
            end
        case 'multiband'
            so = cfg.slicetiming*1000; % specify in ms instead of s
            reference_slice = median(cfg.slicetiming); % this variable should be the timing of the reference slice -> here middle slice
            matlabbatch{1}.spm.temporal.st.ta = 0;
    end
    matlabbatch{1}.spm.temporal.st.so = so;
    matlabbatch{1}.spm.temporal.st.refslice = reference_slice;
    matlabbatch{1}.spm.temporal.st.prefix = 'a';
    spm_jobman('run',matlabbatch)
    
end

%% Create fieldmap

% check if all fieldmaps are there, if not skip the fieldmap related steps
if cfg.do.create_fieldmap

	%plot_sn(i_sub);
	prefix = 'arf';
	include_loc = 0 ; % should a localizer run (or several) be included in preprocessing?

	% Initialize the jobmanager
	spm('defaults','fmri')
	%initjobman; % see function at end of script (to initialize only once)

	% first the first session
	clear matlabbatch

	fieldmap_dir = fullfile(cfg.sub(i_sub).dir,'alldata','fieldmap');
	phase_name = sprintf('^fieldmap%02i-%02i.*\\.(img|nii)',i_sub,4);
	magn_name = sprintf('^fieldmap%02i-%02i.*\\.(img|nii)',i_sub,3);
	phase_name = spm_select('fplist',fieldmap_dir,phase_name);
	magn_name = spm_select('fplist',fieldmap_dir,magn_name);
	magn_name = magn_name(1,:);
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.phase = {[phase_name ',1']};
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.magnitude = {[magn_name ',1']};

	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.et = cfg.fieldmapTE;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.maskbrain = 0 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.blipdir = -1;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.tert = cfg.echo_readout_time;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.epifm = 0 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.ajm = 0 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.uflags.method = 'Mark3D';
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.uflags.fwhm = 10 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.uflags.pad = 0 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.uflags.ws = 1 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.template = {fullfile(cfg.dirs.spm_dir,'toolbox','FieldMap','T1.nii')};
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.fwhm = 5;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.nerode = 2;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.ndilate = 4;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.thresh = 0.5;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.reg = 0.02;

	run_end = cfg.sub(i_sub).import.prf_experiment_runs + ...
		cfg.sub(i_sub).import.uc_experiment_runs;
	for i_run = 1:run_end
		file_dir = fullfile(cfg.sub(i_sub).dir,'alldata',sprintf('run%02i',i_run));
		matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.session(i_run).epi = {fullfile(file_dir,sprintf('%s%02i-%02i-001.%s',prefix,i_sub,i_run,cfg.suffix))};
	end
	if include_loc
		warning('Localizer not included in fieldmap calculation!') %#ok
	end
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.matchvdm = 1 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.sessname = 'session';
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.writeunwarped = 1 ;
	anat_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.anat = {[spm_select('fplist',anat_dir,['^struct',num2str(i_sub,'%02i'),'.*\.(img|nii)']) ',1']};
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.matchanat = 0 ;

	spm_jobman('run',matlabbatch)

	close all

	% then the second session
	clear matlabbatch

	fieldmap_dir = fullfile(cfg.sub(i_sub).dir,'alldata','fieldmap_second');
	phase_name = sprintf('^fieldmap_second%02i-%02i.*\\.(img|nii)',i_sub,4);
	magn_name = sprintf('^fieldmap_second%02i-%02i.*\\.(img|nii)',i_sub,3);
	phase_name = spm_select('fplist',fieldmap_dir,phase_name);
	magn_name = spm_select('fplist',fieldmap_dir,magn_name);
	magn_name = magn_name(1,:);
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.phase = {[phase_name ',1']};
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.magnitude = {[magn_name ',1']};

	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.et = cfg.fieldmapTE;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.maskbrain = 0 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.blipdir = -1;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.tert = cfg.echo_readout_time;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.epifm = 0 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.ajm = 0 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.uflags.method = 'Mark3D';
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.uflags.fwhm = 10 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.uflags.pad = 0 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.uflags.ws = 1 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.template = {fullfile(cfg.dirs.spm_dir,'toolbox','FieldMap','T1.nii')};
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.fwhm = 5;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.nerode = 2;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.ndilate = 4;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.thresh = 0.5;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.defaults.defaultsval.mflags.reg = 0.02;
	ct = 1;
	run_start = cfg.sub(i_sub).import.prf_experiment_runs + ...
		cfg.sub(i_sub).import.uc_experiment_runs + 1;
	run_end = run_start + cfg.sub(i_sub).import.mc_experiment_runs - 1;
	for i_run = run_start:run_end
		file_dir = fullfile(cfg.sub(i_sub).dir,'alldata',sprintf('run%02i',i_run));
		matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.session(ct).epi = {fullfile(file_dir,sprintf('%s%02i-%02i-001.%s',prefix,i_sub,i_run,cfg.suffix))};
		ct = ct+1;
	end
	if include_loc
		warning('Localizer not included in fieldmap calculation!') %#ok
	end
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.matchvdm = 1 ;
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.sessname = 'session';
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.writeunwarped = 1 ;
	anat_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.anat = {[spm_select('fplist',anat_dir,['^struct',num2str(i_sub,'%02i'),'.*\.(img|nii)']) ',1']};
	matlabbatch{1}.spm.tools.fieldmap.presubphasemag.subj.matchanat = 0 ;

	spm_jobman('run',matlabbatch)

	close all

end


%% Apply fieldmap to functional images

if cfg.do.apply_fieldmap

	%plot_sn(i_sub);
	clear matlabbatch
	spm('defaults','fmri')
	loadtoolbox = 1 ;
	spm_jobman('initcfg')

	run_end = cfg.sub(i_sub).import.prf_experiment_runs + ...
		cfg.sub(i_sub).import.uc_experiment_runs;
	for i_run = 1:run_end
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.data.scans = cellstr(spm_select('fplist',fullfile(cfg.sub(i_sub).dir,'alldata',sprintf('run%02i',i_run)),'^arf.*\.(img|nii)$'));
		fn = spm_select('fplist',fullfile(cfg.sub(i_sub).dir,'alldata','fieldmap'),['^vdm.*session' num2str(i_run) '\.(img|nii)$']);
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.data.vdmfile = {[fn ',1']};
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.pedir = 2;
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.which = [2 0];
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.rinterp = 4;
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.wrap = [0 0 0];
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.mask = 0 ;
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.prefix = 'u';
		spm_jobman('run',matlabbatch)
	end
	ct = 1;
	run_start = cfg.sub(i_sub).import.prf_experiment_runs + ...
		cfg.sub(i_sub).import.uc_experiment_runs + 1;
	run_end = run_start + cfg.sub(i_sub).import.mc_experiment_runs - 1;
	for i_run = run_start:run_end
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.data.scans = cellstr(spm_select('fplist',fullfile(cfg.sub(i_sub).dir,'alldata',sprintf('run%02i',i_run)),'^arf.*\.(img|nii)$'));
		fn = spm_select('fplist',fullfile(cfg.sub(i_sub).dir,'alldata','fieldmap_second'),['^vdm.*session' num2str(ct) '.(img|nii)$']);
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.data.vdmfile = {[fn ',1']};
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.pedir = 2;
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.which = [2 0];
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.rinterp = 4;
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.wrap = [0 0 0];
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.mask = 0 ;
		matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.prefix = 'u';
		spm_jobman('run',matlabbatch)
		ct = ct+1;
	end
end

if cfg.do.apply_fieldmap_mean

	%plot_sn(i_sub);
	clear matlabbatch
	spm('defaults','fmri')
	loadtoolbox = 1 ;
	spm_jobman('initcfg')
	prefix = 'meanf'; % prefix of mean image % 'f'

	mean_dir = fullfile(cfg.sub(i_sub).dir,'alldata','mean');
	ref_path = spm_select('fplist',mean_dir,['^' prefix '.*\.(nii|img)$']);

	matlabbatch{1}.spm.tools.fieldmap.applyvdm.data.scans = cellstr(ref_path);
	fn = spm_select('fplist',fullfile(cfg.sub(i_sub).dir,'alldata','fieldmap'),['^vdm.*session' num2str(1) '\.(img|nii)$']);
	matlabbatch{1}.spm.tools.fieldmap.applyvdm.data.vdmfile = {[fn ',1']};
	matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.pedir = 2;
	matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.which = [2 0];
	matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.rinterp = 4;
	matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.wrap = [0 0 0];
	matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.mask = 0 ;
	matlabbatch{1}.spm.tools.fieldmap.applyvdm.roptions.prefix = 'u';
	spm_jobman('run',matlabbatch)
end


%% Do Coregistration

if cfg.do.coreg
    
    %plot_sn(i_sub);
    spm('defaults','fmri')
    %initjobman; % Initialize the jobmanager
    
    prefix = 'umeanf';
    struct_prefix = ['struct',num2str(i_sub,'%02i')];
    
    mean_dir = fullfile(cfg.sub(i_sub).dir,'alldata','mean');
    ref_path = spm_select('fplist',mean_dir,['^' prefix '.*\.(nii|img)$']);
    %if isempty(ref_path), ref_path=spm_select('fplist',mean_dir,['^meanf.*\.(nii|img)$']); end,
    %ref_path = ref_path(1,:);
    %if size(ref_path,1)>1, error('More than 1 mean image in %s',mean_dir), end
    ref_path = [ref_path ',1'];
    
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    source_path = spm_select('fplist',struct_dir,['^' struct_prefix '.*\.(nii|img)$']);
    %if size(source_path,1)>1, error('More than 1 structural image in %s',struct_dir), end
    source_path = [source_path ',1'];
    
    clear matlabbatch
    
    matlabbatch{1}.spm.spatial.coreg.estimate.ref = {ref_path};
    matlabbatch{1}.spm.spatial.coreg.estimate.source = {source_path};
    matlabbatch{1}.spm.spatial.coreg.estimate.other = {''};
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.cost_fun = 'nmi';
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.sep = [4 2];
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.tol = [0.02 0.02 0.02 0.001 0.001 0.001 0.01 0.01 0.01 0.001 0.001 0.001];
    matlabbatch{1}.spm.spatial.coreg.estimate.eoptions.fwhm = [7 7];
    spm_jobman('run',matlabbatch)
    
end


%% For each run, merge 3D .nii volumes to a single 4D file

if cfg.do.nii_3Dto4D

    include_loc = [0 0] ;
    spm('defaults','fmri')
    %initjobman; % Initialize the jobmanager
    
    prefix = 'uarf'; % 'arf'
    cfg.suffix = 'nii';

    clear matlabbatch
    
    files_all = select_files_adapted(cfg,i_sub,prefix,include_loc);
	
    for f=1:length(files_all)
        files = files_all{f};
        subind = sprintf('sub%02d',i_sub);
        sub_dir = fullfile(cfg.dirs.derived_dir,subind);
        runind = sprintf('run%02d',f);
        run_dir = fullfile(sub_dir,'alldata',runind);
		save_name = fullfile(run_dir, [prefix num2str(i_sub,'%02.f') '-' num2str(f,'%02.f') '.nii']);
        nii_3Dto4D(files,save_name)
    end	

end


%% Create the mean .nii file for each run

if cfg.do.nii_mean

    runs = cfg.sub(i_sub).import.prf_experiment_runs + ...
        cfg.sub(i_sub).import.uc_experiment_runs + ...
        cfg.sub(i_sub).import.mc_experiment_runs;
    prefix = 'uarf'; % 'arf'
    subind = sprintf('sub%02d',i_sub);
    sub_dir = fullfile(cfg.dirs.derived_dir,subind);
        for r=1:runs
            runind = sprintf('run%02d',r);
            run_dir = fullfile(sub_dir,'alldata',runind);
            nifti_dir = fullfile(run_dir, [prefix num2str(i_sub,'%02.f') '-' num2str(r,'%02.f') '.nii']);
            nii_info = spm_vol(nifti_dir);
            imageData = zeros(nii_info(1).dim);
            for vd=1:length(nii_info)
                volData = spm_read_vols(nii_info(vd));
                imageData = imageData + volData;
            end
            meanData = imageData / numel(nii_info);
            meanVol = nii_info(1);
            meanVol.fname = fullfile(run_dir, [prefix num2str(i_sub,'%02.f') '-' num2str(r,'%02.f') '_mean.nii']);
            spm_write_vol(meanVol, meanData);
        end

end


%% Do Segmentation and write normalized anatomy

if cfg.do.segment
    
    %plot_sn(i_sub);
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    
    struct_prefix = ['struct',num2str(i_sub)];
    
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    struct_path = spm_select('fplist',struct_dir,['^' struct_prefix '.*\.(img|nii)$']);
    struct_path = [struct_path ',1'];
    
    clear matlabbatch
    
    matlabbatch{1}.spm.spatial.preproc.channel.vols = {struct_path};
    matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001 ;
    matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 60 ;
    matlabbatch{1}.spm.spatial.preproc.channel.write = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm = {[cfg.dirs.spm_dir,'/tpm/TPM.nii,1']};
    matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus = 1 ;
    matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 1];
    matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm = {[cfg.dirs.spm_dir,'/tpm/TPM.nii,2']};
    matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus = 1 ;
    matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 1];
    matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm = {[cfg.dirs.spm_dir,'/tpm/TPM.nii,3']};
    matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 1];
    matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm = {[cfg.dirs.spm_dir,'/tpm/TPM.nii,4']};
    matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus = 3;
    matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm = {[cfg.dirs.spm_dir,'/tpm/TPM.nii,5']};
    matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus = 4;
    matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [1 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm = {[cfg.dirs.spm_dir,'/tpm/TPM.nii,6']};
    matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus = 2;
    matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [0 0];
    matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];
    matlabbatch{1}.spm.spatial.preproc.warp.mrf = 1 ;
    matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1 ;
    matlabbatch{1}.spm.spatial.preproc.warp.reg = [0 0.001 0.5 0.05 0.2];
    matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
    matlabbatch{1}.spm.spatial.preproc.warp.fwhm = 0 ;
    matlabbatch{1}.spm.spatial.preproc.warp.samp = 2;
    matlabbatch{1}.spm.spatial.preproc.warp.write = [1 1];
    
    % Normalize anatomical (to check with template how well normalization
    % worked)
    matlabbatch{2}.spm.spatial.normalise.write.subj.def(1) = cfg_dep('Segment: Forward Deformations', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('.','fordef', '()',{':'}));
    matlabbatch{2}.spm.spatial.normalise.write.subj.resample = {struct_path};
    %[bbox, vox] = spm_get_bbox(struct_path);
    matlabbatch{2}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{2}.spm.spatial.normalise.write.woptions.vox = [1 1 1];
    matlabbatch{2}.spm.spatial.normalise.write.woptions.interp = 4;
    
    spm_jobman('run',matlabbatch)
    
end

%% Check visually if segmentation worked

if cfg.do.check_segment
    
    seg_prefix = 'c';
    segment_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    segment_path = spm_select('fplist',segment_dir,['^' seg_prefix '.*\.(img|nii)$']);
    segment_path = [segment_path repmat(',1',size(segment_path,1),1)];
    
    spm_check_registration(segment_path)
    
    print(fullfile(fullfile(cfg.sub(i_sub).dir,'alldata','other'), ['check_segment.jpg']), ...
        '-djpeg', '-r300')
end

%% smooth functional images

if cfg.do.smoothing
    
    %plot_sn(i_sub);
    only_loc = 1; %specify if only localizer images should be smoothed or all images
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    
    prefix = 'arf'; % 'rf'
    
    cfg.suffix = '(nii|img)';
    
    clear matlabbatch
    
    if only_loc
        fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^', prefix, 'loc.*\.(nii|img)$']));
        fnames
    else
        fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^', prefix, '.*\.(nii|img)$']));
    end
    
    FWHM = cfg.FWHM;
    
    if ~isempty(fnames)
        
        loadtoolbox = 1 ;
        spm_jobman('initcfg')
        %plot_sn(i_sub);
        clear matlabbatch
        
        % smooth
        matlabbatch{1}.spm.spatial.smooth.data = fnames;
        matlabbatch{1}.spm.spatial.smooth.fwhm = [FWHM FWHM FWHM];
        matlabbatch{1}.spm.spatial.smooth.dtype = 0 ;
        matlabbatch{1}.spm.spatial.smooth.im = 0 ;
        matlabbatch{1}.spm.spatial.smooth.prefix = sprintf('s%02d',FWHM);
        spm_jobman('run',matlabbatch)
        
    end
end

%% normalize functional images + WM and CSF files for estimation of the noise components

if cfg.do.normalize
    
    %plot_sn(i_sub);
    only_loc = 0 ;
    only_struct = 1;
    
    spm('defaults','fmri')
    initjobman; %
    prefix = 'arf'; % 'rf'
    % check if all fieldmaps are there
    if ~isempty(cfg.sub(i_sub).import.fieldmap) && ~isempty(cfg.sub(i_sub).import.second_fieldmap), prefix='uarf';end
    
    cfg.suffix = '(nii|img)';
    
    if only_loc
        fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata','localizer'),['^', prefix, '.*\.(nii|img)$']));
    elseif ~only_loc
        fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^', prefix, '.*\.(nii|img)$']));
    end
    
    if only_struct
        fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata','struct'),['^(c3|c2).*\.(nii|img)$']));
    end
    
    % normalize
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    normparams_path = spm_select('fplist',fullfile(struct_dir),['^y_.*\.(nii|img)$']); %path to forward transformation file
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = fnames;
    [bbox, vox_sz] = spm_get_bbox(fnames{1});
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = vox_sz ;%[2 2 2];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 2;
    
    spm_jobman('run',matlabbatch)
end
%% tapas denoising


if cfg.do.tapas_denoising
    
    %plot_sn(i_sub);
    include_loc = [0 0];
    spm('defaults','fmri')
    initjobman; %
    
    %create folder for tapas if it does not exist
    if ~isdir(fullfile(cfg.sub(i_sub).dir,'tapas')), mkdir(fullfile(cfg.sub(i_sub).dir,'tapas')), end
    
    % add tapas to path
    if ismac
        addpath(genpath('/Users/pablo/Documents/phd/RecCOR/fMRI/analysis_tools/tapas-master'))
    elseif isunix
        addpath('/scratch/singej96/dfg_projekt/WP1/analysis_tools/tapas-master')
    end
    tapas_init();
    % remove old tapas version
    %rmpath(genpath('/afs/cbs.mpg.de/software/spm/toolboxes/tapas-physio/2.4.0.0'));
    prefix = 'arf'; % 'rf'
    % check if all fieldmaps are there
    if ~isempty(cfg.sub(i_sub).import.fieldmap) && ~isempty(cfg.sub(i_sub).import.second_fieldmap), prefix='uarf';end
    
    cfg.suffix = '(nii|img)';
    
    functionals = select_files_adapted(cfg,i_sub,prefix,include_loc);
    
    
    noise_rois = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata','struct'),['^(c3|c2).*\.(nii|img)$']));
    
    for run = 1:cfg.sub(i_sub).n_runs
        clear matlabbatch
        matlabbatch{1}.spm.tools.physio.save_dir = {''};
        matlabbatch{1}.spm.tools.physio.log_files.vendor = 'Philips';
        matlabbatch{1}.spm.tools.physio.log_files.cardiac = {''};
        matlabbatch{1}.spm.tools.physio.log_files.respiration = {''};
        matlabbatch{1}.spm.tools.physio.log_files.scan_timing = {''};
        matlabbatch{1}.spm.tools.physio.log_files.sampling_interval = [];
        matlabbatch{1}.spm.tools.physio.log_files.relative_start_acquisition = 0;
        matlabbatch{1}.spm.tools.physio.log_files.align_scan = 'last';
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nslices = 39;
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.NslicesPerBeat = [];
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.TR = 1;
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Ndummies = 0;
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nscans = 470;
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.onset_slice = ceil(39/2);
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.time_slice_to_slice = [];
        matlabbatch{1}.spm.tools.physio.scan_timing.sqpar.Nprep = [];
        matlabbatch{1}.spm.tools.physio.scan_timing.sync.nominal = struct([]);
        matlabbatch{1}.spm.tools.physio.preproc.cardiac.modality = 'ECG';
        matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.min = 0.4;
        matlabbatch{1}.spm.tools.physio.preproc.cardiac.initial_cpulse_select.auto_matched.file = 'initial_cpulse_kRpeakfile.mat';
        matlabbatch{1}.spm.tools.physio.preproc.cardiac.posthoc_cpulse_select.off = struct([]);
        matlabbatch{1}.spm.tools.physio.model.output_multiple_regressors = fullfile(cfg.sub(i_sub).dir,'tapas',['tapas_regressors_check_run',num2str(run,'%02.f'),'.txt']);
        matlabbatch{1}.spm.tools.physio.model.output_physio = fullfile(cfg.sub(i_sub).dir,'tapas',['physio_run',num2str(run,'%02.f'),'.mat']);
        matlabbatch{1}.spm.tools.physio.model.orthogonalise = 'all';
        matlabbatch{1}.spm.tools.physio.model.censor_unreliable_recording_intervals = false;
        matlabbatch{1}.spm.tools.physio.model.retroicor.no = struct([]);
        matlabbatch{1}.spm.tools.physio.model.rvt.no = struct([]);
        matlabbatch{1}.spm.tools.physio.model.hrv.no = struct([]);
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.fmri_files = functionals{run};
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.roi_files = noise_rois;
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.force_coregister = {'no'};
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.thresholds = 0.1; %% parameters set here are corresponding to the recommendations in the CONN-toolbox documentation % parameters before 0.6
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.n_voxel_crop = 1;
        matlabbatch{1}.spm.tools.physio.model.noise_rois.yes.n_components = 5; % parameter before 3
        run_dir = fullfile(cfg.sub(i_sub).dir,'alldata',sprintf('run%02d',run));
        matlabbatch{1}.spm.tools.physio.model.movement.yes.file_realignment_parameters = {spm_select('fplist',run_dir,'^rp_.*\.txt$')};
        matlabbatch{1}.spm.tools.physio.model.movement.yes.order = 24;
        matlabbatch{1}.spm.tools.physio.model.movement.yes.movement_censoring_method = 'none';
        matlabbatch{1}.spm.tools.physio.model.movement.yes.outlier_translation_mm = Inf;
        matlabbatch{1}.spm.tools.physio.model.movement.yes.outlier_rotation_deg = Inf;
        matlabbatch{1}.spm.tools.physio.model.other.no = struct([]);
        %matlabbatch{1}.spm.tools.physio.model.other.yes.input_multiple_regressors = {spm_select('fplist',run_dir,'^rp_.*\.txt$');};
        matlabbatch{1}.spm.tools.physio.verbose.level = 1;
        matlabbatch{1}.spm.tools.physio.verbose.fig_output_file = fullfile(cfg.sub(i_sub).dir,'tapas',['tapas_output_run',num2str(run,'%02.f'),'.jpg']);
        matlabbatch{1}.spm.tools.physio.verbose.use_tabs = false;
        spm_jobman('run',matlabbatch)
    end
end

%% GLM denoise

if cfg.do.glmdenoise
    % add GLMdenoise to path
    addpath('/data/pt_02348/objdraw/fmri/GLMdenoise-master')
    setup;
    
    include_loc = 0;
    prefix = 'arf'; % 'rf'
    cfg.suffix = '(nii|img)';
    resname = 'first_level_denoise';
    outname = 'tryout';
    
    standard_glmdenoise(cfg,i_sub,resname,prefix,outname);
    
end

%% GLM single

if cfg.do.glmsingle
    % add GLMdenoise to path
    addpath(genpath('/data/pt_02348/objdraw/fmri/GLMsingle'))
    addpath(genpath('/data/pt_02348/objdraw/fmri/fracridge'))
    
    setup;
    
    include_loc = 0;
    prefix = 'arf'; % 'rf'
    cfg.suffix = '(nii|img)';
    resname = 'GLMsingle';
    outdir = fullfile(cfg.sub(i_sub).dir,'results','GLM',resname);
    
    %create onsets files to create design matrix
    onset_type = 'round';
    [design,conditions] = create_design_from_onsets(cfg, i_sub,onset_type);
    
    
    
    standard_glmsingle(cfg,i_sub,prefix,design,conditions,outdir);
    
end

%% ==========================================================
%% FIRSTLEVEL ANALYSES
%% ==========================================================

%% Firstlevel full (includes most relevant variables)

if cfg.do.firstlevel_localizer
    
    %plot_sn(i_sub);
    prefix = 's05arf';
    onsname = 'localizer'; % name of onsets file
    resname = 'first_level_localizer_both_sessions'; % results path
    onset_type = 'exact'; % exact or round image onsets
    fir = 0;
    is_loc = 1;
    loc_count = ~isempty(cfg.sub(i_sub).import.localizer) + ~isempty(cfg.sub(i_sub).import.second_localizer);
    mparams = 1 ; % do / do not include movement parameters
    tapas_denoise = 0; % do/ do not include tapas noise regressors
    physio = 0 ; % do / do not include physiological variables
    glmdenoise =  0;
    global hrf_fitting
    hrf_fitting = 0; % specify if hrf fitting should be applied
    
    if hrf_fitting ==1
        
        addpath(genpath(cfg.dirs.glmsingle_dir))
    end
    
    % write onsets file
    if strcmpi(onsname, 'cat_onsets')
        create_onset_files_categories(cfg, i_sub,cfg.sub(i_sub).pid,onset_type);
    elseif strcmpi(onsname, 'localizer') % use other function to create onsets when button presses should be taken als onsets (control analysis)
        create_localizer_onset_vdm(cfg, i_sub,loc_count);
    end
    
    if ~cfg.do.contrasts_only
        matlabbatch = standard_firstlevel(cfg,i_sub,prefix,onsname,resname,mparams,physio,tapas_denoise,glmdenoise,fir,is_loc);
        spm_jobman('run',matlabbatch)
    end
    
    results_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM',resname);
    
    if cfg.do.first_level_contrasts
        
        % Set contrasts
        clear matlabbatch
        if strcmpi(onsname, 'cat_onsets')
            matlabbatch{1}.spm.stats.con.spmmat = {fullfile(results_dir,'SPM.mat')};
            matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'im_on';
            matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = [ones(1,60)];
            matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';
            matlabbatch{2}.spm.stats.con.spmmat = {fullfile(results_dir,'SPM.mat')};
            matlabbatch{2}.spm.stats.con.consess{1}.tcon.name = 'manmade_vs_natural';
            matlabbatch{2}.spm.stats.con.consess{1}.tcon.weights = [ones(1,30) ones(1,30)*-1];
            matlabbatch{2}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';
        elseif strcmpi(onsname, 'localizer') % use different contrast for button presses (control analysis)
            matlabbatch{1}.spm.stats.con.spmmat = {fullfile(results_dir,'SPM.mat')};
            matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'objects_bigger_scrambled';
            matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = [1 -1 0];
            matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';
            matlabbatch{2}.spm.stats.con.spmmat = {fullfile(results_dir,'SPM.mat')};
            matlabbatch{2}.spm.stats.con.consess{1}.tcon.name = 'scenes_bigger_objects';
            matlabbatch{2}.spm.stats.con.consess{1}.tcon.weights = [-1 0 1];
            matlabbatch{2}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';
            matlabbatch{3}.spm.stats.con.spmmat = {fullfile(results_dir,'SPM.mat')};
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'all_bigger_baseline';
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = [1 1 1];
            matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';
            
        end
        spm_jobman('run',matlabbatch)
        
    end
    
end
%% First level without HRF fitting


if cfg.do.firstlevel_no_hrf_fitting
    
    %plot_sn(i_sub);
    prefix = 'arf';
    % check if all fieldmaps are there
    if ~isempty(cfg.sub(i_sub).import.fieldmap) && ~isempty(cfg.sub(i_sub).import.second_fieldmap), prefix='uarf';end
    onsname = 'cat_onsets'; % name of onsets
    onset_type = 'exact'; % exact or round image onsets
    fir = 0;
    is_loc = 0;
    mparams = 0 ; % do / do not include movement parameters
    tapas_denoise = 1; % do/ do not include tapas noise regressors
    physio = 0 ; % do / do not include physiological variables
    glmdenoise =  0;
    global hrf_fitting
    hrf_fitting = 0; % specify if hrf fitting should be applied
    
    % write onsets file
    if strcmpi(onsname, 'cat_onsets')
        create_onset_files_pilot(cfg, i_sub,cfg.sub(i_sub).pid,onset_type);
    elseif strcmpi(onsname, 'button_onsets') % use other function to create onsets when button presses should be taken als onsets (control analysis)
        create_onset_files_buttonpress(cfg, i_sub);
    end
    
    resname = 'no_hrf_fitting_explicit_mask';
    
    if ~cfg.do.contrasts_only
        matlabbatch = standard_firstlevel(cfg,i_sub,prefix,onsname,resname,mparams,physio,tapas_denoise,glmdenoise,fir,is_loc);
        spm_jobman('run',matlabbatch)
    end
    
    if cfg.do.first_level_contrasts
        
        % setup condition names - for identifying the images in the SPM.mat
        condition_names = cell(1,242);
        for i=1:242
            condition_names(i) = {['Image_', num2str(i)]}; %image names here
        end
        
        % load info about which images are control and which
        % challenge
        load('labelinfo.mat')
        control_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='control');
        control_condition_names = condition_names(labelinfo.sorted_cat =='control');
        challenge_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='challenge');
        challenge_condition_names = condition_names(labelinfo.sorted_cat =='challenge');
        
        
        % now create contrast weights manually by looping through blocks
        results_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM',resname);
        load(fullfile(results_dir,'SPM.mat'))
        control_weights = [];
        challenge_weights = [];
        difference_weights = [];
        all_bigger_baseline_weights = [];
        nuissance_weights = zeros(1,36);
        for run = 1:length(SPM.Sess)
            
            this_conds = cellfun(@char,{SPM.Sess(run).U.name},'UniformOutput',false);
            this_weights = ismember(this_conds,control_condition_names);
            control_weights = [control_weights, this_weights,nuissance_weights];
            this_weights = ismember(this_conds,challenge_condition_names);
            challenge_weights = [challenge_weights, this_weights,nuissance_weights];
            
        end
        difference_weights =control_weights*-1+challenge_weights;
        all_bigger_baseline_weights = control_weights+challenge_weights;
        
        
        %Set contrasts
        clear matlabbatch
        if strcmpi(onsname, 'cat_onsets')
            matlabbatch{1}.spm.stats.con.spmmat = {fullfile(results_dir,'SPM.mat')};
            matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'control';
            matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = control_weights./(length(SPM.Sess)/2);
            matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
            matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'challenge';
            matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = challenge_weights./(length(SPM.Sess)/2);
            matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
            matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'challenge vs. control';
            matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = difference_weights./length(SPM.Sess);
            matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
            matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'all vs. baseline';
            matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = all_bigger_baseline_weights/length(SPM.Sess);
            matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
            matlabbatch{1}.spm.stats.con.delete = 1;
            
        end
        spm_jobman('run',matlabbatch)
        
    end
end
%% Firstlevel with HRF fitting

if cfg.do.firstlevel_hrf_fitting
    
    %plot_sn(i_sub);
    prefix = 'arf';
    % check if all fieldmaps are there
    if ~isempty(cfg.sub(i_sub).import.fieldmap) && ~isempty(cfg.sub(i_sub).import.second_fieldmap), prefix='uarf';end
    onsname = 'cat_onsets'; % name of onsets
    onset_type = 'exact'; % exact or round image onsets
    fir = 0;
    is_loc = 0;
    mparams = 0 ; % do / do not include movement parameters
    tapas_denoise = 1; % do/ do not include tapas noise regressors
    physio = 0 ; % do / do not include physiological variables
    glmdenoise =  0;
    global hrf_fitting
    hrf_fitting = 1; % specify if hrf fitting should be applied
    
    if hrf_fitting ==1
        
        addpath(genpath(cfg.dirs.glmsingle_dir))
    end
    
    % write onsets file
    if strcmpi(onsname, 'cat_onsets')
        create_onset_files_pilot(cfg, i_sub,cfg.sub(i_sub).pid,onset_type);
    elseif strcmpi(onsname, 'button_onsets') % use other function to create onsets when button presses should be taken als onsets (control analysis)
        create_onset_files_buttonpress(cfg, i_sub);
    end
    
    % setup condition names - for identifying the images in the SPM.mat
    condition_names = cell(1,242);
    for i=1:242
        condition_names(i) = {['Image_', num2str(i)]}; %image names here
    end
    
    % load info about which images are control and which
    % challenge
    load('labelinfo.mat')
    control_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='control');
    control_condition_names = condition_names(labelinfo.sorted_cat =='control');
    challenge_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='challenge');
    challenge_condition_names = condition_names(labelinfo.sorted_cat =='challenge');
    
    global hrf_idx
    
    for hrf_idx = 1:20
        global hrf_fitting
        resname = num2str(hrf_idx); % results path
        %results_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM',resname);
        results_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting',resname);
        
        %check if glm was already finished
        %if exist(fullfile(results_dir,'con0004.nii')),continue;end %residual name Res_7520
        
        fprintf('Running HRF %02i\n',hrf_idx);
        if ~cfg.do.contrasts_only
            matlabbatch = standard_firstlevel(cfg,i_sub,prefix,onsname,resname,mparams,physio,tapas_denoise,glmdenoise,fir,is_loc);
            spm_jobman('run',matlabbatch)
        end
        
        
        % now create contrast weights manually by looping through blocks
        if cfg.do.first_level_contrasts
            run_contrasts_rcor(results_dir,control_condition_names,challenge_condition_names);
        end
        
    end
    
    if cfg.do.eval_hrf_fitting
        
        results_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM', 'hrf_fitting');
        
        % find best fitting HRF for each voxel and write betas and residuals
        % with the best fitting HRF for each voxel
        eval_hrf_fitting_optimized(results_dir,'fitted');
        
    end
end

if cfg.do.only_eval_hrf_fitting
    
    results_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting');
    
    % find best fitting HRF for each voxel and write betas and residuals
    % with the best fitting HRF for each voxel
    eval_hrf_fitting_optimized(results_dir,'fitted_hrflibrary_check');
    
end


%% Normalize first-level betas for searchlight analyses

if cfg.do.normalize_first_level
    
    %plot_sn(i_sub);
    
    spm('defaults','fmri')
    initjobman; %
    cfg.suffix = '(nii|img)';
    
    first_level_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted_explicit_brainmask');
    %     if ~isfolder(first_level_dir)
    %         first_level_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted');
    %     end
    %
    % first clean up all normalized files in the directory (in case there
    % are any)
    files = dir(fullfile(first_level_dir, 'w*')); % find all files in the folder that start with 'w'
    files
    for i = 1:length(files)
        filename = fullfile(first_level_dir, files(i).name); % get the full file path
        delete(filename); % delete the file
    end
    
    % then get filenames for the remaining non normalized betas and
    % residuals
    fnames = cellstr(spm_select('fplistrec',first_level_dir,['\.', cfg.suffix,'$']));
    
    % normalize
    clear matlabbatch
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    normparams_path = spm_select('fplist',fullfile(struct_dir),['^y_.*\.(nii|img)$']); %path to forward transformation file
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = fnames;
    %[bbox, vox_sz] = spm_get_bbox(fnames{1});
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 2;
    
    spm_jobman('run',matlabbatch)
    
    % move the normalized files to a new directory
    
    if isdir(fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','normalized_explicit_brainmask')); rmdir(fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','normalized_explicit_brainmask'),'s');
    elseif ~isdir(fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','normalized_explicit_brainmask')); mkdir(fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','normalized_explicit_brainmask')); end
    
    cd(fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted_explicit_brainmask'))
    movefile('*w*', fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','normalized_explicit_brainmask'));
    copyfile('SPM.mat', fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','normalized_explicit_brainmask'));
    
end

%% ROI definition

% Early Visual Cortex ROI definition

if cfg.do.write_ROI_EVC
    
    %plot_sn(i_sub);
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    
    %create folder for rois if it does not exist
    if ~isdir(fullfile(cfg.sub(i_sub).dir,'roi')), mkdir(fullfile(cfg.sub(i_sub).dir,'roi')), end
    
    % now transfer normalized EVC anatomical definition into subject
    % space
    cfg.suffix = '(nii|img)';
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    clear matlabbatch
    % setup some specifics for the transformation
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    normparams_path = spm_select('fplist',fullfile(struct_dir),['^iy_.*\.(nii|img)$']); %path to inverse transformation file
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
    mask_path = fullfile(cfg.dirs.derived_dir,'roi','evc_mni.nii');
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {mask_path};
    % get a functional hdr for defining the bounding box and the voxel size
    fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^umeanfloc.*\.(nii|img)$']));
    if isempty(fnames{1}),  fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^meanfloc.*\.(nii|img)$'])); end
    func_hdr = spm_vol(fnames{1});
    [bbox,vox] = spm_get_bbox(func_hdr,'fv');
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = bbox;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = vox;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'indiv_';
    
    disp('Transforming EVC mask from MNI into subject space')
    spm_jobman('run',matlabbatch)
    disp('Done.');
    
    %move file into subject directory
    movefile(fullfile(cfg.dirs.derived_dir,'roi','indiv_evc_mni.nii'),fullfile(cfg.sub(i_sub).dir,'roi'),'f');
    %rename file
    movefile(fullfile(cfg.sub(i_sub).dir,'roi','indiv_evc_mni.nii'),fullfile(cfg.sub(i_sub).dir,'roi','evcmask.nii'));
    
    clear matlabbatch
    matlabbatch{1}.spm.spatial.coreg.write.ref = {fnames{1}};
    matlabbatch{1}.spm.spatial.coreg.write.source = {fullfile(cfg.sub(i_sub).dir,'roi','evcmask.nii')};
    matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 1;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = '';
    
    disp('Resampling individual EVC mask to functional space')
    spm_jobman('run',matlabbatch)
    
    % get the overlap of EVC mask and all vs. baseline contrast
    write_ROI_mask_EVC(cfg, i_sub);
    disp('Done.')
    
end

% LOC ROI definition

if cfg.do.write_ROI_LOC
    
    %plot_sn(i_sub);
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    
    %create folder for rois if it does not exist
    if ~isdir(fullfile(cfg.sub(i_sub).dir,'roi')), mkdir(fullfile(cfg.sub(i_sub).dir,'roi')), end
    
    % now transfer normalized EVC anatomical definition into subject
    % space
    cfg.suffix = '(nii|img)';
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    clear matlabbatch
    % setup some specifics for the transformation
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    normparams_path = spm_select('fplist',fullfile(struct_dir),['^iy_.*\.(nii|img)$']); %path to inverse transformation file
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
    mask_path = fullfile(cfg.dirs.derived_dir,'roi','loc_mni.nii');
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {mask_path};
    % get a functional hdr for defining the bounding box and the voxel size
    fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^umeanfloc.*\.(nii|img)$']));
    if isempty(fnames{1}),  fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^meanfloc.*\.(nii|img)$'])); end
    func_hdr = spm_vol(fnames{1});
    [bbox,vox] = spm_get_bbox(func_hdr,'fv');
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = bbox;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = vox;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'indiv_';
    
    disp('Transforming LOC mask from MNI into subject space')
    spm_jobman('run',matlabbatch)
    disp('Done.');
    
    %move file into subject directory
    movefile(fullfile(cfg.dirs.derived_dir,'roi','indiv_loc_mni.nii'),fullfile(cfg.sub(i_sub).dir,'roi'),'f');
    %rename file
    movefile(fullfile(cfg.sub(i_sub).dir,'roi','indiv_loc_mni.nii'),fullfile(cfg.sub(i_sub).dir,'roi','loc_mask.nii'));
    
    clear matlabbatch
    matlabbatch{1}.spm.spatial.coreg.write.ref = {fnames{1}};
    matlabbatch{1}.spm.spatial.coreg.write.source = {fullfile(cfg.sub(i_sub).dir,'roi','loc_mask.nii')};
    matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 1;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = '';
    
    disp('Resampling individual LOC mask to functional space')
    spm_jobman('run',matlabbatch)
    
    % get the overlap of EVC mask and all vs. baseline contrast
    write_ROI_mask_LOC(cfg, i_sub);
    disp('Done.')
    
end

% PPA ROI definition

if cfg.do.write_ROI_PPA
    
    %plot_sn(i_sub);
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    
    %create folder for rois if it does not exist
    if ~isdir(fullfile(cfg.sub(i_sub).dir,'roi')), mkdir(fullfile(cfg.sub(i_sub).dir,'roi')), end
    
    % now transfer normalized EVC anatomical definition into subject
    % space
    cfg.suffix = '(nii|img)';
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    clear matlabbatch
    % setup some specifics for the transformation
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    normparams_path = spm_select('fplist',fullfile(struct_dir),['^iy_.*\.(nii|img)$']); %path to inverse transformation file
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
    mask_path = fullfile(cfg.dirs.derived_dir,'roi','ppa_mni.nii');
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {mask_path};
    % get a functional hdr for defining the bounding box and the voxel size
    fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^umeanfloc.*\.(nii|img)$']));
    if isempty(fnames{1}),  fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^meanfloc.*\.(nii|img)$'])); end
    func_hdr = spm_vol(fnames{1});
    %     [bbox,vox] = spm_get_bbox(func_hdr,'fv');
    %     matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = bbox;
    %     matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = vox;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [1 1 1];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'indiv_';
    
    disp('Transforming IT mask from MNI into subject space')
    spm_jobman('run',matlabbatch)
    disp('Done.');
    
    %move file into subject directory
    movefile(fullfile(cfg.dirs.derived_dir,'roi','indiv_ppa_mni.nii'),fullfile(cfg.sub(i_sub).dir,'roi'),'f');
    %rename file
    movefile(fullfile(cfg.sub(i_sub).dir,'roi','indiv_ppa_mni.nii'),fullfile(cfg.sub(i_sub).dir,'roi','PPA_mask.nii'));
    
    clear matlabbatch
    matlabbatch{1}.spm.spatial.coreg.write.ref = {fnames{1}};
    matlabbatch{1}.spm.spatial.coreg.write.source = {fullfile(cfg.sub(i_sub).dir,'roi','PPA_mask.nii')};
    matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 1;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = '';
    
    disp('Resampling individual PPA mask to functional space')
    spm_jobman('run',matlabbatch)
    
    % get the overlap between the PPA mask and the scenes > other contrast
    write_ROI_mask_PPA(cfg,i_sub);
    disp('Done.')
end

if cfg.do.check_ROI_overlap
    
    disp('Checking for overlap between ROI masks and removing any overlapping voxels')
    check_ROI_overlap(cfg,i_sub)
end

if cfg.do.write_brain_mask
    
    %plot_sn(i_sub);
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    
    %create folder for rois if it does not exist
    if ~isdir(fullfile(cfg.sub(i_sub).dir,'roi')), mkdir(fullfile(cfg.sub(i_sub).dir,'roi')), end
    
    % now transfer normalized EVC anatomical definition into subject
    % space
    cfg.suffix = '(nii|img)';
    spm('defaults','fmri')
    initjobman; % Initialize the jobmanager
    clear matlabbatch
    % setup some specifics for the transformation
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    normparams_path = spm_select('fplist',fullfile(struct_dir),['^iy_.*\.(nii|img)$']); %path to inverse transformation file
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
    mask_path = fullfile(cfg.dirs.derived_dir,'roi','brainmask_canlab.nii');
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {mask_path};
    % get a functional hdr for defining the bounding box and the voxel size
    fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^umeanfloc.*\.(nii|img)$']));
    if isempty(fnames{1}),  fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^meanfloc.*\.(nii|img)$'])); end
    
    func_hdr = spm_vol(fnames{1});
    [bbox,vox] = spm_get_bbox(func_hdr,'fv');
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = bbox;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = vox;
    %matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
    %    78 76 85];
    %matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [1 1 1];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'indiv_';
    
    disp('Transforming brain mask from MNI into subject space')
    spm_jobman('run',matlabbatch)
    
    %move file into subject directory
    movefile(fullfile(cfg.dirs.derived_dir,'roi','indiv_brainmask_canlab.nii'),fullfile(cfg.sub(i_sub).dir,'roi'),'f');
    %rename file
    movefile(fullfile(cfg.sub(i_sub).dir,'roi','indiv_brainmask_canlab.nii'),fullfile(cfg.sub(i_sub).dir,'roi','brainmask.nii'));
    
    clear matlabbatch
    matlabbatch{1}.spm.util.imcalc.input = cellstr(fullfile(cfg.sub(i_sub).dir,'roi','brainmask.nii'));
    matlabbatch{1}.spm.util.imcalc.output = fullfile(cfg.sub(i_sub).dir,'roi','brainmask.nii');
    matlabbatch{1}.spm.util.imcalc.outdir = {''};
    matlabbatch{1}.spm.util.imcalc.expression = 'i1>0.1';
    matlabbatch{1}.spm.util.imcalc.var = struct('name', {}, 'value', {});
    matlabbatch{1}.spm.util.imcalc.options.dmtx = 0;
    matlabbatch{1}.spm.util.imcalc.options.mask = 0;
    matlabbatch{1}.spm.util.imcalc.options.interp = 1;
    matlabbatch{1}.spm.util.imcalc.options.dtype = 4;
    spm_jobman('run',matlabbatch)
    disp('Done.');
end

%% Decoding Analyses

if cfg.do.unn
    
    for i=1:242
        condition_names(i) = {['Image_', num2str(i)]}; % here there should be your image names
    end
    labels = [1:242];
    beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM', 'hrf_fitting','fitted');
    beta_avg_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','unn');
    apply_unn_betas(condition_names, beta_dir, beta_avg_dir, cfg,i_sub);
end

if cfg.do.decoding_obj % 4 runs
    
    clear cfgd
    cfgd.analysis = 'searchlight';
    cfgd.noisenorm = 1; % specify if multivariate noise normalization should be applied
    cfgd.hrf_fitting = 1; % specify if hrf fitting betas should be used for loading the residuals
    cfgd.parallel = 0;
    avg_size = 2; % how many betas to average into one beta
    condition_names = cell(1,242);
    for i=1:242
        condition_names(i) = {['Image_', num2str(i)]}; % here there should be your image names
    end
    load('labelinfo.mat')
    control_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='control');
    control_condition_names = condition_names(labelinfo.sorted_cat =='control');
    challenge_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='challenge');
    challenge_condition_names = condition_names(labelinfo.sorted_cat =='challenge');
    beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM', 'hrf_fitting','fitted');
    beta_avg_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','avg');
    control_out_dir = fullfile(cfg.sub(i_sub).dir,'results','decoding','obj_decoding_control',cfgd.analysis);
    challenge_out_dir = fullfile(cfg.sub(i_sub).dir,'results','decoding','obj_decoding_challenge',cfgd.analysis);
    roi_dir = fullfile(cfg.sub(i_sub).dir,'roi');
    if strcmpi(cfgd.analysis, 'searchlight')
        cfgd.files.mask = {fullfile(beta_dir,'mask.nii')};
        if contains(beta_dir, 'samevoxsz')
            cfgd.files.mask = {fullfile(cfg.dirs.derived_dir,'normalized_intersect_mask_samevoxsz.nii')};
        end
    elseif strcmpi(cfgd.analysis, 'roi')
        cfgd.files.mask = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'loc_mask.nii');fullfile(roi_dir, 'PPA_mask.nii')};
    end
    if cfg.do.avg_betas;
        avg_betas(condition_names,n_runs/avg_size, beta_dir, beta_avg_dir,cfg.sub(i_sub).n_betas, cfg);
    end
    decoding_nobetas(control_condition_names,control_labels,beta_dir,control_out_dir,cfgd,cfg,i_sub);
    decoding_nobetas(challenge_condition_names,challenge_labels,beta_dir,challenge_out_dir,cfgd,cfg,i_sub);
end


if cfg.do.decoding_obj_nobetas % 4 runs
    
    clear cfgd
    cfgd.analysis = 'roi';
    cfgd.parallel = 0; % if decoding should be parallelized
    cfgd.noisenorm = 0; % specify if multivariate noise normalization should be applied
    cfgd.decoding.software = 'libsvm'; % specify if libsvm or liblinear should be used
    cfgd.hrf_fitting = 0; % specify if hrf fitting betas should be used for loading the residuals
    cfgd.perm = 0; %if crossvalidation should be multiplied to 50 folds
    avg_size = 1; % only one mean beta
    n_runs = cfg.sub(i_sub).n_betas; % how many runs are there for a given subject
    condition_names = cell(1,242);
    for i=1:242
        condition_names(i) = {['Image_', num2str(i)]}; %image names here
    end
    
    beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted');
    beta_avg_dir = fullfile(beta_dir, 'avg');
    roi_dir = fullfile(cfg.sub(i_sub).dir,'roi');
    cfgd.design.function.name = 'make_design_cv';
    %cfgd.results.output = {'decision_values'};%, 'accuracy_matrix'};
    if strcmpi(cfgd.analysis, 'searchlight')
        cfgd.files.mask = {fullfile(beta_dir,'mask.nii')};
        if contains(beta_dir, 'samevoxsz')
            cfgd.files.mask = {fullfile(cfg.dirs.derived_dir,'normalized_intersect_mask_samevoxsz.nii')};
        end
    elseif strcmpi(cfgd.analysis, 'roi')
        cfgd.files.mask = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'PPA_mask.nii');fullfile(roi_dir, 'loc_mask.nii')};
    end
    
    % ugly solution to missing SPM.mat
    if ~isfile(fullfile(beta_dir, 'SPM.mat'))
        copyfile(fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted','SPM.mat'), fullfile(beta_dir, 'SPM.mat'))
    end
    
    if cfg.do.avg_betas && cfgd.noisenorm ==1; noisenorm_avg_betas_split(condition_names,avg_size, beta_dir, beta_avg_dir,n_runs, cfgd.files.mask,i_sub, cfg);
    elseif cfg.do.avg_betas && cfgd.noisenorm ==0; avg_betas(condition_names,avg_size, beta_dir, beta_avg_dir,n_runs, cfg); end
    
    % set labels and condition names for challenge and control
    load('labelinfo.mat')
    control_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='control')+1;
    control_condition_names = condition_names(labelinfo.sorted_cat =='control');
    challenge_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='challenge')+1;
    challenge_condition_names = condition_names(labelinfo.sorted_cat =='challenge');
    % set output paths
    control_out_dir = fullfile(cfg.sub(i_sub).dir,'results','decoding','obj_decoding_control',cfgd.analysis);
    challenge_out_dir = fullfile(cfg.sub(i_sub).dir,'results','decoding','obj_decoding_challenge',cfgd.analysis);
    decoding_obj_nobetas(control_condition_names,control_labels,beta_avg_dir,control_out_dir,cfgd,i_sub);
    decoding_obj_nobetas(challenge_condition_names,challenge_labels,beta_avg_dir,challenge_out_dir,cfgd,i_sub);
    combine_obj_decoding_results(control_out_dir,max(control_labels));
    combine_obj_decoding_results(challenge_out_dir,max(challenge_labels));
    
end

if cfg.do.crossdecoding_obj_nobetas
    
    clear cfgd
    cfgd.analysis = 'searchlight';
    cfgd.parallel = 0; % if decoding should be parallelized
    cfgd.noisenorm = 0; % specify if multivariate noise normalization should be applied
    cfgd.decoding.software = 'libsvm'; % specify if libsvm or liblinear should be used
    cfgd.hrf_fitting = 0; % specify if hrf fitting betas should be used for loading the residuals
    cfgd.perm = 0; %if crossvalidation should be multiplied to 50 folds
    avg_size = 1; % only one mean beta
    n_runs = cfg.sub(i_sub).n_betas; % how many runs are there for a given subject
    condition_names = cell(1,242);
    for i=1:242
        condition_names(i) = {['Image_', num2str(i)]}; %image names here
    end
    
    beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted');
    beta_avg_dir = fullfile(beta_dir, 'avg');
    roi_dir = fullfile(cfg.sub(i_sub).dir,'roi');
    cfgd.design.function.name = 'make_design_cv';
    %cfgd.results.output = {'decision_values'};%, 'accuracy_matrix'};
    if strcmpi(cfgd.analysis, 'searchlight')
        cfgd.files.mask = {fullfile(beta_dir,'mask.nii')};
        if contains(beta_dir, 'samevoxsz')
            cfgd.files.mask = {fullfile(cfg.dirs.derived_dir,'normalized_intersect_mask_samevoxsz.nii')};
        end
    elseif strcmpi(cfgd.analysis, 'roi')
        cfgd.files.mask = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'PPA_mask.nii');fullfile(roi_dir, 'loc_mask.nii')};
    end
    
    % ugly solution to missing SPM.mat
    if ~isfile(fullfile(beta_dir, 'SPM.mat'))
        copyfile(fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted','SPM.mat'), fullfile(beta_dir, 'SPM.mat'))
    end
    
    if cfg.do.avg_betas && cfgd.noisenorm ==1; noisenorm_avg_betas_split(condition_names,avg_size, beta_dir, beta_avg_dir,n_runs, cfgd.files.mask,i_sub, cfg);
    elseif cfg.do.avg_betas && cfgd.noisenorm ==0; avg_betas(condition_names,avg_size, beta_dir, beta_avg_dir,n_runs, cfg); end
    
    % set labels and condition names for challenge and control
    load('labelinfo.mat')
    control_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='control')+1;
    control_condition_names = condition_names(labelinfo.sorted_cat =='control');
    challenge_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='challenge')+1;
    challenge_condition_names = condition_names(labelinfo.sorted_cat =='challenge');
    % set output paths
    out_dir = fullfile(cfg.sub(i_sub).dir,'results','decoding','obj_crossdecoding',cfgd.analysis);
    crossdecoding_obj_nobetas(control_condition_names,challenge_condition_names,control_labels,challenge_labels,beta_avg_dir,out_dir,cfgd,i_sub);
    combine_obj_decoding_results(out_dir,max(control_labels));
end


if cfg.do.decoding_nobetas_controversial_congruent
    
    clear cfgd
    cfgd.analysis = 'searchlight';
    cfgd.noisenorm = 0; % specify if multivariate noise normalization should be applied
    cfgd.hrf_fitting = 0; % specify if hrf fitting betas should be used for loading the residuals
    avg_size = 1; % how many betas to average into one beta
    condition_names = cell(1,90);
    for i=1:60
        condition_names(i) = {['Image_', num2str(i)]};
    end
    
    ct = 1;
    for i =1:60
        
        condition_names(ct+60) = {['Controversial_Image_', num2str(i)]};
        ct = ct+1;
    end
    
    beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM', 'hrf_fitting','fitted');
    beta_avg_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','avg');
    controversial_beta_avg_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','avg_controversial');
    out_dir = fullfile(cfg.sub(i_sub).dir,'results','decoding','crossdecoding_controversial_hrf_fitting_only_congruent',cfgd.analysis);
    roi_dir = fullfile(cfg.sub(i_sub).dir,'roi');
    if strcmpi(cfgd.analysis, 'searchlight')
        cfgd.files.mask = {fullfile(beta_dir,'mask.nii')};
        if contains(beta_dir, 'samevoxsz')
            cfgd.files.mask = {fullfile(cfg.dirs.derived_dir,'normalized_intersect_mask_samevoxsz.nii')};
        end
    elseif strcmpi(cfgd.analysis, 'roi')
        cfgd.files.mask = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'loc_mask.nii');fullfile(roi_dir, 'PPA_mask.nii')};
    end
    if cfg.do.avg_betas;
        avg_betas_controversial(condition_names(length(condition_names)/2+1:end),beta_dir, controversial_beta_avg_dir, cfg);
        avg_betas(condition_names(1:length(condition_names)/2),avg_size, beta_dir, beta_avg_dir,cfg.sub(i_sub).n_betas, cfg);
    end
    crossdecoding_controversial_congruent(condition_names,cfg.sub(i_sub).n_betas/avg_size,beta_avg_dir,controversial_beta_avg_dir, out_dir,cfgd);
    %run("evaluate_decision_vals_controversial_stimuli.m");
end

if cfg.do.decoding_nobetas_controversial_congruent_manmade
    
    clear cfgd
    cfgd.analysis = 'searchlight';
    cfgd.noisenorm = 0; % specify if multivariate noise normalization should be applied
    cfgd.hrf_fitting = 0; % specify if hrf fitting betas should be used for loading the residuals
    avg_size = 2; % how many betas to average into one beta
    condition_names = cell(1,90);
    for i=1:60
        condition_names(i) = {['Image_', num2str(i)]};
    end
    
    ct = 1;
    for i =1:60
        
        condition_names(ct+60) = {['Controversial_Image_', num2str(i)]};
        ct = ct+1;
    end
    
    beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM', 'hrf_fitting','fitted');
    beta_avg_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','avg');
    controversial_beta_avg_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','avg_controversial');
    out_dir = fullfile(cfg.sub(i_sub).dir,'results','decoding','crossdecoding_controversial_hrf_fitting_only_congruent_manmade',cfgd.analysis);
    roi_dir = fullfile(cfg.sub(i_sub).dir,'roi');
    if strcmpi(cfgd.analysis, 'searchlight')
        cfgd.files.mask = {fullfile(beta_dir,'mask.nii')};
        if contains(beta_dir, 'samevoxsz')
            cfgd.files.mask = {fullfile(cfg.dirs.derived_dir,'normalized_intersect_mask_samevoxsz.nii')};
        end
    elseif strcmpi(cfgd.analysis, 'roi')
        cfgd.files.mask = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'loc_mask.nii');fullfile(roi_dir, 'PPA_mask.nii')};
    end
    if cfg.do.avg_betas;
        avg_betas_controversial(condition_names(length(condition_names)/2+1:end),beta_dir, controversial_beta_avg_dir, cfg);
        avg_betas(condition_names(1:length(condition_names)/2),avg_size, beta_dir, beta_avg_dir,cfg.sub(i_sub).n_betas, cfg);
    end
    crossdecoding_controversial_congruent_manmade(condition_names,cfg.sub(i_sub).n_betas/avg_size,beta_avg_dir,controversial_beta_avg_dir, out_dir,cfgd);
    %run("evaluate_decision_vals_controversial_stimuli.m");
end



if cfg.do.decoding_nobetas_only_controversial
    
    clear cfgd
    cfgd.analysis = 'searchlight';
    cfgd.noisenorm = 0; % specify if multivariate noise normalization should be applied
    cfgd.hrf_fitting = 0; % specify if hrf fitting betas should be used for loading the residuals
    avg_size = 2; % how many betas to average into one beta
    condition_names = cell(1,60);
    
    for i =1:60
        
        condition_names(i) = {['Controversial_Image_', num2str(i)]}
    end
    beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM', 'hrf_fitting','fitted');
    beta_avg_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','avg');
    controversial_beta_avg_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','avg_controversial');
    out_dir = fullfile(cfg.sub(i_sub).dir,'results','decoding','crossdecoding_only_controversial_hrf_fitting',cfgd.analysis);
    roi_dir = fullfile(cfg.sub(i_sub).dir,'roi');
    if strcmpi(cfgd.analysis, 'searchlight')
        cfgd.files.mask = {fullfile(beta_dir,'mask.nii')};
        if contains(beta_dir, 'samevoxsz')
            cfgd.files.mask = {fullfile(cfg.dirs.derived_dir,'normalized_intersect_mask_samevoxsz.nii')};
        end
    elseif strcmpi(cfgd.analysis, 'roi')
        cfgd.files.mask = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'loc_mask.nii');fullfile(roi_dir, 'PPA_mask.nii')};
    end
    if cfg.do.avg_betas;
        avg_betas(condition_names,avg_size, beta_dir, controversial_beta_avg_dir,length(cfg.sub(i_sub).import.second_experiment), cfg);
    end
    crossdecoding_only_controversial(condition_names,cfg.sub(i_sub).n_betas/avg_size,beta_avg_dir,controversial_beta_avg_dir, out_dir,cfgd);
    %run("evaluate_decision_vals_controversial_stimuli.m");
end


%% RSA

if cfg.do.decoding_similarity_pearson % 4 runs
    
    clear cfgd
    cfgd.analysis = 'roi';
    cfgd.hrf_fitting = 1;
    condition_names = cell(1,242);
    cfgd.noisenorm = 1;
    for i=1:242
        condition_names(i) = {['Image_', num2str(i)]}; % here there should be your image names
    end
    load('labelinfo.mat')
    control_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='control');
    control_condition_names = condition_names(labelinfo.sorted_cat =='control');
    challenge_labels = labelinfo.sorted_obj_nr(labelinfo.sorted_cat =='challenge');
    challenge_condition_names = condition_names(labelinfo.sorted_cat =='challenge');
    beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted');
    roi_dir = fullfile(cfg.sub(i_sub).dir,'roi');
    cfgd.design.function.name = 'make_design_similarity';
    if strcmpi(cfgd.analysis, 'searchlight')
        cfgd.files.mask = {fullfile(beta_dir,'mask.nii')};
        if contains(beta_dir, 'normalized')
            cfgd.files.mask = {fullfile(cfg.dirs.derived_dir,'normalized_intersect_mask.nii')};
        end
        if contains(beta_dir, 'samevoxsz')
            cfgd.files.mask = {fullfile(cfg.dirs.derived_dir,'normalized_intersect_mask_samevoxsz.nii')};
        end
    elseif strcmpi(cfgd.analysis, 'roi')
        cfgd.files.mask = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'PPA_mask.nii');fullfile(roi_dir, 'loc_mask.nii')};
    end
    control_out_dir = fullfile(cfg.sub(i_sub).dir,'results','RSA_hrf_fitting_noisenorm',cfgd.analysis,'control');
    challenge_out_dir = fullfile(cfg.sub(i_sub).dir,'results','RSA_hrf_fitting_noisenorm',cfgd.analysis,'challenge');
    decoding_similarity_pearson(control_condition_names,control_labels,beta_dir,control_out_dir,cfgd);
    decoding_similarity_pearson(challenge_condition_names,challenge_labels,beta_dir,challenge_out_dir,cfgd);
    
end

if cfg.do.decoding_similarity_crossnobis % 4 runs
    
    clear cfgd
    cfgd.analysis = 'roi';
    conds = {'Photo'; 'Drawing'; 'Sketch'};
    
    for this_cond_ind = 1:length(conds)
        condition_names = cell(1,48);
        cond = conds{this_cond_ind};
        for i=1:48
            condition_names(i) = {[cond, '_', num2str(i)]};
        end
        labels = [1:48];
        cfgd.beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','first_level_check');
        out_dir = fullfile(cfg.sub(i_sub).dir,'results','RSA_crossnobis_tapas_check',cfgd.analysis,cond);
        roi_dir = fullfile(cfg.sub(i_sub).dir,'roi');
        if strcmpi(cfgd.analysis, 'searchlight')
            cfgd.files.mask = {fullfile(cfgd.beta_dir,'mask.nii')};
            if contains(cfgd.beta_dir, 'normalized')
                cfgd.files.mask = {fullfile(cfg.dirs.derived_dir,'normalized_intersect_mask.nii')};
            end
        elseif strcmpi(cfgd.analysis, 'roi')
            cfgd.files.mask = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'combined_loc_fus_mask.nii')}; %fullfile(roi_dir, 'rIT_mask.nii')};
        end
        decoding_similarity_crossnobis(condition_names,labels,cfgd.beta_dir,out_dir,cfgd);
    end
end


if cfg.do.decoding_similarity_all % 4 runs
    
    clear cfgd
    cfgd.analysis = 'roi';
    conds = {'Photo'; 'Drawing'; 'Sketch'};
    condition_names = [];
    for this_cond_ind = 1:length(conds)
        cond = conds{this_cond_ind};
        for i=1:48
            cond_names(i) = {[cond, '_', num2str(i)]};
        end
        condition_names = cat(2,condition_names,cond_names);
    end
    labels = 1:length(condition_names);
    beta_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','first_level_denoise');
    out_dir = fullfile(cfg.sub(i_sub).dir,'results','BIG_RSA_final',cfgd.analysis,'all');
    roi_dir = fullfile(cfg.sub(i_sub).dir,'roi');
    cfgd.design.function.name = 'make_design_similarity';
    if strcmpi(cfgd.analysis, 'searchlight')
        cfgd.files.mask = {fullfile(beta_dir,'mask.nii')};
    elseif strcmpi(cfgd.analysis, 'roi')
        cfgd.files.mask = {fullfile(roi_dir, 'evcmask.nii');fullfile(roi_dir, 'combined_loc_fus_mask.nii')};
    end
    decoding_similarity_pearson(condition_names,labels,beta_dir,out_dir,cfgd);
end

%% smooth and normalize results

if cfg.do.smooth_norm_res
    
    FWHM = cfg.FWHM;
    
    %select files to normalize and smooth here
    loc_fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted_hrflibrary_check'),'^spmT'));
    first_level_fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting','fitted_hrflibrary_check'),'^con'));
    fnames = cat(1,loc_fnames,first_level_fnames);
    %new_res_dir = fullfile(cfg.sub(i_sub).dir,'results','decoding','obj_crossdecoding');
    %     if ~isdir(new_res_dir), mkdir(new_res_dir), end
    %     for idx=1:length(res_names)
    %         copyfile(res_names{idx},new_res_dir)
    %     end
    %fnames = cellstr(spm_select('fplistrec',new_res_dir,'^mean'));
    fnames
    if ~isempty(fnames)
        
        loadtoolbox = 1 ;
        spm_jobman('initcfg')
        %plot_sn(i_sub);
        clear matlabbatch
        
        %normalize
        struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
        normparams_path = spm_select('fplist',fullfile(struct_dir),['^y_.*\.(nii|img)$']); %path to forward transformation file
        matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
        matlabbatch{1}.spm.spatial.normalise.write.subj.resample = fnames;
        matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
            78 76 85];
        matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
        matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 2;
        
        % smooth
        matlabbatch{2}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
        matlabbatch{2}.spm.spatial.smooth.fwhm = [FWHM FWHM FWHM];
        matlabbatch{2}.spm.spatial.smooth.dtype = 0 ;
        matlabbatch{2}.spm.spatial.smooth.im = 0 ;
        matlabbatch{2}.spm.spatial.smooth.prefix = sprintf('s%02d',FWHM);
        spm_jobman('run',matlabbatch)
        
    end
end


%% prepare first level results for usage with the GSS approach

if cfg.do.norm_mask_for_GSS
    
    FWHM = cfg.FWHM;
    
    %select files to normalize here - first_level results and localizer
    %results
    loc_fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'results','GLM','no_hrf_fitting_explicit_mask'),'^spmT*\.(nii|img)$'));
    first_level_fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'results','GLM','no_hrf_fitting_explicit_mask'),'^con*\.(nii|img)$'));
    
    loadtoolbox = 1 ;
    spm_jobman('initcfg')
    %%plot_sn(i_sub);
    clear matlabbatch
    
    % normalize
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    normparams_path = spm_select('fplist',fullfile(struct_dir),['^y_.*\.(nii|img)$']); %path to forward transformation file
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = loc_fnames;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 2;
    spm_jobman('run',matlabbatch)
    
    clear matlabbatch
    
    % normalize
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    normparams_path = spm_select('fplist',fullfile(struct_dir),['^y_.*\.(nii|img)$']); %path to forward transformation file
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = first_level_fnames;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = [-78 -112 -70
        78 76 85];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = [2 2 2];
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 2;
    spm_jobman('run',matlabbatch)
    
    
    % get a binary mask from the localizer contrast using the function
    % from the spm_ss toolbox
    loc_norm_fname = fullfile(cfg.sub(i_sub).dir,'results','GLM','localizer','wspmT_0001.nii');
    outpath = fullfile(cfg.sub(i_sub).dir,'results','GLM','localizer');
    createlocalizermask(loc_norm_fname, outpath);
    
    
end

%% SUBFUNCTIONS
function matlabbatch = standard_firstlevel(cfg,i_sub,prefix,onsname,resname,mparams,physio,tapas_denoise,glmdenoise,fir,is_loc)
n_slices = cfg.n_slices;

results_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM',resname);

global hrf_fitting

if hrf_fitting == 1
    results_dir = fullfile(cfg.sub(i_sub).dir,'results','GLM','hrf_fitting',resname); % different directory structure for the HRF fitting
end

if ~isdir(results_dir), mkdir(results_dir), end
if exist(fullfile(results_dir,'SPM.mat'),'file') && cfg.do.firstlevel_overwrite == 0
    error('SPM.mat already exists for %s',results_dir)
else
    spm_unlink(fullfile(results_dir,'SPM.mat')); % delete SPM to overwrite without dialog
end

spm('defaults','fmri')
initjobman; % Initialize the jobmanager

% Set up model
if exist('is_loc', 'var') && is_loc ==1
    matlabbatch = firstlevel_localizer(cfg,prefix,results_dir,onsname,i_sub,mparams,physio,glmdenoise,n_slices);
elseif exist('fir','var') && fir == 1
    %matlabbatch = firstlevel_fir(cfg,prefix,results_dir,onsname,i_sub,mparams,physio,glmdenoise,n_slices);
else
    matlabbatch = firstlevel(cfg,prefix,results_dir,onsname,i_sub,mparams,physio,tapas_denoise,glmdenoise,n_slices);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function initjobman

persistent loaded
global loadtoolbox

loadtoolbox = 0 ; %#ok<NASGU> % do not load SPM toolboxes to save time

if isempty(loaded)
    try
        disp('Initializing job manager...')
        spm_jobman('initcfg');
        disp('done.')
    catch %#ok<CTCH>
        disp(lasterr) %#ok<LERR>
        loadtoolbox = 1 ;
        error(['Error loading the SPM jobmanager. Maybe a different ',...
            'version of SPM was running previously. Run ''clear classes'' ',...
            'and try again. Otherwise restart Matlab.'])
    end
    loaded = 1 ;
end

loadtoolbox = 1 ;

function plot_sn(i_sub)
fprintf('Running sub %i...\n',i_sub)

