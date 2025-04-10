% function make_dicom_paths(cfg)
% This function makes the dicom paths where data of each subject should be 
% copied. The location is specified by cfg. If cfg doesn't exist, it will
% be called by config_subjects.

function make_dicom_paths(cfg)

if ~exist('cfg','var')
    cfg = config_subjects;
end

dicom_dir = fullfile(cfg.dirs.data_dir,'DICOM');

for i_sub = cfg.subject_indices
    
    sub_id_dir = cfg.sub(i_sub).id;

    new_dir = fullfile(dicom_dir,sub_id_dir);
    
    if ~isdir(new_dir), mkdir(new_dir), end
        
end












