% function [hdr,cfg] = convert_nii_to_spm_visdecmak(nii,json,cfg,i_sub)
%
% This function takes the filenames of the converted nifti and json files and converts
% them into files that can be used by SPM and saves them in a folder
% structure that is suitable for all further analyses. The
% structure can be set by calling config_subjects (see config_subject.m,
% for an example that you can modify). If the field PatientID doesn't
% exist, it is added from cfg.sub(i_sub).pid.
%
% cfg is a struct and needs the following fields:
%    sub: One sub-structure for each subject, with at least the field
%       import: Specifies the numbers that the scanner gave to each run,
%               with the following fields including some examples:
%           anatomy = 2
%           field_map = [3 4]
%           experiment = [5 6 8 9 10]
%           localizer = 11 % this is a functional localizer run
%    dirs: Directory structure with fields:
%       data_dir: Basic directory of all data of all subjects (e.g. c:\myexperiment)
%       sub_dir(i_sub): Structure containing the subject directories of
%           all subjects (optional, but useful later)
%    n_runs: Number of experimental runs of the current participant
%    n_dummy: Number of dummy scans in the beginning that were not
%       automatically discarded by the scanner
%    n_scans_experiment: Number of scans of the experiment (optional)
%    n_scans_localizer: Number of scans of the localizer (optional)
%    prefix: Name of your experiment, will end up in the file names (e.g.
%       myexperiment)
%    import: Specifies the numbers that the scanner gave to each run, with
%       the following fields including some examples:
%       anatomy = 2
%       field_map = [3 4]
%       experiment = [5 6 8 9 10]
%       localizer = 11 % this is a functional localizer run
%
%
%   If this doesn't fit your description, because e.g. you have several
%   localizers, you need to change this function to fit to your needs.
%
% Update 2013/07/26: Automatically discard unnecessary images, introduce
%   trash folder for these images
% 2009, by Martin N. Hebart

function convert_nii_to_spm_visdecmak(nii_anat,json_anat, nii_fmap, ...
	json_fmap, nii_func, json_func, input_dir, cfg, i_sub)

anatomy_name = 'struct';
fieldmap_name = 'fieldmap';
other_name = 'other';
trash_name = 'trash';
localizer_name = 'localizer';
parameters_name = 'parameters';
derived_dir = cfg.dirs.derived_dir;

%% Set paths in the standard manner

sub_dir = fullfile(derived_dir,sprintf('sub%02d',i_sub),'alldata');

mkdirc(fullfile(sub_dir,anatomy_name))
mkdirc(fullfile(sub_dir,fieldmap_name))
mkdirc(fullfile(sub_dir,other_name))
mkdirc(fullfile(sub_dir,trash_name))
mkdirc(fullfile(sub_dir,parameters_name))
mkdirc(fullfile(sub_dir, localizer_name))
mkdirc(fullfile(sub_dir,'mean'))

n_runs = cfg.sub(i_sub).n_runs;

for i_run = 1:n_runs
    run_dir = sprintf('run%02d',i_run);
    mkdirc(fullfile(sub_dir,run_dir))
end



%% Extract anatomicals

% load the json file
fname = fullfile(input_dir, 'anat', json_anat{1});
fid = fopen(fname);
raw = fread(fid,inf);
str = char(raw');
fclose(fid);
hdr = jsondecode(str);

% Get the anatomical images
fname = sprintf('%s%.2d-%.2d.img',...
	anatomy_name,i_sub,hdr.SeriesNumber);
anat_hdr = spm_vol(fullfile(input_dir, 'anat', nii_anat{1}));
anat_vol = spm_read_vols(anat_hdr);
anat_hdr.fname = fullfile(sub_dir,anatomy_name,fname);
spm_write_vol(anat_hdr,anat_vol);



%% Extract filedmaps

for i = 1:length(nii_fmap)
	
	% load the json file
	fname = fullfile(input_dir, 'fmap', json_fmap{i});
	fid = fopen(fname);
	raw = fread(fid,inf);
	str = char(raw');
	fclose(fid);
	hdr = jsondecode(str);
	
	% Get the fieldmap images
	fname = sprintf('%s%.2d-%.2d-%s-%i.nii', fieldmap_name, i_sub, ...
		hdr.SeriesNumber, hdr.ImageType{3}, hdr.EchoTime*100000);
	fieldmap_hdr = spm_vol(fullfile(input_dir, 'fmap', nii_fmap{i}));
	fieldmap_vol = spm_read_vols(fieldmap_hdr);
	fieldmap_hdr.fname = fullfile(sub_dir,fieldmap_name,fname);
	spm_write_vol(fieldmap_hdr, fieldmap_vol);

end



%% Extract functional
runs_functional = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];

for i = 1:length(nii_func)
	
	% load the json file
	fname = fullfile(input_dir, 'func', json_func{i});
	fid = fopen(fname);
	raw = fread(fid,inf);
	str = char(raw');
	fclose(fid);
	hdr = jsondecode(str);
	
	% Get the functional images
	i_run = runs_functional(i);

	% split the 4D nifti to 3D nifits in the functional folder
	spm_file_split(fullfile(input_dir, 'func', nii_func{i}),fullfile(sub_dir,sprintf('run%.2d',i_run)));

	% get all images in the folder
	func_files = dir(fullfile(sub_dir,sprintf('run%.2d',i_run),'*.nii'));
	func_files = {func_files.name}'

	for i_image = 1:length(func_files)

		% write the image
		fname = sprintf('f%s%.2d-%.2d-%.3d.nii',...
			cfg.prefix,i_sub,i_run,i_image);
		fname = fullfile(sub_dir,sprintf('run%.2d',i_run),fname);
		movefile(fullfile(sub_dir,sprintf('run%.2d',i_run),func_files{i_image}),fname)

	end

end



%%%%%%%%%%%%%%%%%%%%%
function mkdirc(d)
    if exist(d,'dir'), delete(fullfile(d,'*')), 
elseif ~exist(d,'dir'), mkdir(d), end