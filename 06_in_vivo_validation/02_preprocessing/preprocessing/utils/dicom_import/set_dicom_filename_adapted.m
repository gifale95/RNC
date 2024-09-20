% function [hdr,cfg] = set_dicom_filename(hdr,cfg,i_sub)
% 
% This function introduces the new field "fname" into each loaded DICOM
% header which will be the filename specified by the structure cfg. The
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

function [hdr,cfg] = set_dicom_filename(hdr,cfg,i_sub)

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

n_runs = cfg.sub(i_sub).n_runs;
    
for i_run = 1:n_runs
    run_dir = sprintf('run%02d',i_run);
    mkdirc(fullfile(sub_dir,run_dir))
end

%% get the indices of all aquisition steps

fnames = cell(length(hdr),1);

for i = 1:length(hdr) % go through all DICOMs and add field fname to hdr
    
    % If it's an anatomical image
    if contains(hdr{i}.ProtocolName, cfg.sub(i_sub).import.anatomy)
        fname = sprintf('%s%.2d-%.2d-%.3d.img',...
            anatomy_name,i_sub,hdr{i}.SeriesNumber,hdr{i}.InstanceNumber);
        hdr{i}.fname = fullfile(sub_dir,anatomy_name,fname);
        hdr{i}.prefix = 's';
        
    % If it's a functional run    
    elseif sum(ismember(cfg.sub(i_sub).import.experiment,hdr{i}.SeriesNumber))
        i_run = find(cfg.sub(i_sub).import.experiment == hdr{i}.SeriesNumber);
        i_image = hdr{i}.InstanceNumber;
        i_image = i_image - cfg.n_dummy;
        
        if i_image >= 1 && i_image <= cfg.n_scans_experiment
            fname = sprintf('%s%.2d-%.2d-%.3d.img',...
                cfg.prefix,i_sub,i_run,i_image);
            hdr{i}.fname = fullfile(sub_dir,sprintf('run%.2d',i_run),fname);
        elseif i_image <= 0
            fname = sprintf('%s%.2d-%.2d-dummy%.3d.img',...
                cfg.prefix,i_sub,i_run,hdr{i}.InstanceNumber);
            hdr{i}.fname = fullfile(sub_dir,trash_name,fname);
        else
            fname = sprintf('%s%.2d-%.2d-%.3d.img',...
                cfg.prefix,i_sub,i_run,i_image);
            hdr{i}.fname = fullfile(sub_dir,trash_name,fname);
        end
        hdr{i}.prefix = 'f';
        
    % If it's a localizer run    
    elseif sum(ismember(cfg.sub(i_sub).import.localizer,hdr{i}.SeriesNumber))
        i_run = find(cfg.sub(i_sub).import.localizer == hdr{i}.SeriesNumber);
        i_image = hdr{i}.InstanceNumber;
        i_image = i_image - cfg.n_dummy;
        if i_image >= 1 && i_image <= cfg.n_scans_localizer
            fname = sprintf('%s%.2d-%.2d-%.3d.img',...
                cfg.prefix_localizer,i_sub,i_run,i_image);
            hdr{i}.fname = fullfile(sub_dir,'localizer',fname);
        elseif i_image <0
            fname = sprintf('%s%.2d-%.2d-dummy%.3d.img',...
                cfg.prefix_localizer,i_sub,i_run,hdr{i}.InstanceNumber);
            hdr{i}.fname = fullfile(sub_dir,trash_name,fname);
        else
            fname = sprintf('%s%.2d-%.2d-%.3d.img',...
                cfg.prefix_localizer,i_sub,i_run,i_image);
            hdr{i}.fname = fullfile(sub_dir,trash_name,fname);
        end
        hdr{i}.prefix = 'f';
        
    % If fieldmap
    elseif sum(ismember(cfg.sub(i_sub).import.fieldmap,hdr{i}.SeriesNumber))
        fname = sprintf('%s%.2d-%.2d-%.3d-%.2d.img',...
            fieldmap_name,i_sub,hdr{i}.SeriesNumber,hdr{i}.InstanceNumber,hdr{i}.EchoNumbers);
        hdr{i}.fname = fullfile(sub_dir,fieldmap_name,fname);
       
    % Otherwise    
    else
        fname = sprintf('%s%.2d-%.2d-%.3d.img',...
            cfg.prefix,i_sub,hdr{i}.SeriesNumber,hdr{i}.InstanceNumber);
        hdr{i}.fname = fullfile(sub_dir,other_name,fname);
        hdr{i}.prefix = 's';
    end
    
    fnames{i} = hdr{i}.fname;
    
    % if field PatientID doesn't exist, add it
    if ~isfield(hdr{i},'PatientID')
        hdr{i}.PatientID = cfg.sub(i_sub).pid;
    end
    
end

% Run check that no Number exists twice
fnames_length = max(cell2mat(cellfun(@(x) length(x), fnames, 'uniformOutput',false))); % maximal filename length
fnames = cell2mat(cellfun(@(x) [x repmat(' ',1,fnames_length - length(x))], fnames, 'uniformOutput',false));

for i = 2:size(fnames,1)
    if strcmp(fnames(i,:),fnames(i-1,:))
        error('Double Entries found in the filenames of subject %i (e.g. %s). Please check!',i_sub,fnames(i,:))
    end
end


%%%%%%%%%%%%%%%%%%%%%
function mkdirc(d)

if ~exist(d,'dir'), mkdir(d), end