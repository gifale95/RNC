% function  check_ROI_overlap(cfg, i_sub, p, df)
%
% This code can be used to check for an overlap between previously defined
% ROI masks (EVC and LOC+Pfus) 
% We do a number of things:
% 1. We load the subject-specific EVC and LOC and PPA mask
% 2. We get the overlap between the ROI masks 
% 3. We discard all the voxels that overlap in any of the masks 
% 4. We write the adjusted ROI masks 
%
% INPUT:
%   cfg: a config structure with subject specific and experiment specific
%   details obtained from config_subjects_objdraw.m
%   i_sub: the number of the subject for which the masks should be written
%   struct_path(optional): if path to structural image is given then ROIs
%   are plotted on the T1 of the subject 
%
function check_ROI_overlap(cfg,i_sub,ROI_type,struct_path, ROI_fig_path)

% load PPA mask
try
    PPA_maskhdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['PPA_mask_',ROI_type,'.nii']));
    PPA_mask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['PPA_mask_',ROI_type,'.nii'])));
catch 
    PPA_maskhdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['PPA_mask.nii']));
    PPA_mask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['PPA_mask.nii'])));
end

% load LOC mask
try
    LOC_maskhdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['loc_mask_',ROI_type,'.nii']));
    LOC_mask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['loc_mask_',ROI_type,'.nii'])));
catch
    LOC_maskhdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['loc_mask.nii']));
    LOC_mask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['loc_mask.nii'])));
end 

%load EVC mask
try
    EVC_maskhdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['evcmask_',ROI_type,'.nii']));
    EVC_mask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['evcmask_',ROI_type,'.nii'])));
catch
    EVC_maskhdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['evcmask.nii']));
    EVC_mask = spm_read_vols(spm_vol(fullfile(cfg.sub(i_sub).dir,'roi',['evcmask.nii'])));
end

%get all pairwise overlap

overlap1 = logical(EVC_mask.*LOC_mask); 
overlap2 = logical(EVC_mask.*PPA_mask);
overlap3 = logical(LOC_mask.*PPA_mask); 


% get all voxels that have overlap in any of the pairwise overlaps 
overlap = overlap1 | overlap2 | overlap3 ;

% if overlap is not empty then discard the overlap 
if sum(overlap,'all') > 0 

EVC_mask(overlap ==1) = 0 ;     
LOC_mask(overlap ==1) = 0 ;    
PPA_mask(overlap ==1) = 0 ;

spm_write_vol(EVC_maskhdr,EVC_mask);
spm_write_vol(LOC_maskhdr, LOC_mask); 
spm_write_vol(PPA_maskhdr, PPA_mask); 
end 


% if desired plot the ROIs on the anatomical image
if nargin>2 
    cmapping = colormap('hsv'); 
    fprintf('Plotting ROIs on struct'); 
    plot_roi_on_struct(struct_path, EVC_maskhdr,EVC_mask,LOC_mask,PPA_mask,'', cmapping); 
    
print(ROI_fig_path, ...
              '-djpeg', '-r300')
end 
end 