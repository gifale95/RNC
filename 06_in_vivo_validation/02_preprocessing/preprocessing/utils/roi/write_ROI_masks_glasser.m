function write_ROI_masks_glasser(cfg,i_sub)

%% loop through wang masks, transform them to individual space and then get the overlap with the localizer contrast
spm('defaults','fmri')
initjobman; % Initialize the jobmanager

%create folder for rois if it does not exist
if ~isdir(fullfile(cfg.sub(i_sub).dir,'roi','glasser_masks')), mkdir(fullfile(cfg.sub(i_sub).dir,'roi','glasser_masks')), end

% first load the table with the glasser ROI definitions
glasser_sel = '/scratch/singej96/rcor_collab/derived/roi/Glasser_2016_Table_selection.xlsx';
glasser_sel = readtable(glasser_sel);

glasser_indices = glasser_sel.ParcelIndex;

% now transfer normalized EVC anatomical definition into subject
% space
cfg.suffix = '(nii|img)';
spm('defaults','fmri')
initjobman; % Initialize the jobmanager

% set path
glasser_dir = '/scratch/singej96/rcor_collab/derived/roi/glasser_roi_masks';

% now loop over wang ROIs
for roi = 1:length(glasser_indices)
    
    % first get ROI mask from glasser atlas in glasser atlas space
    disp('Write ROI mask in MNI space')
    % write mask for V1-V3 (still in MNI space)
    ROI = glasser_indices(roi);
    this_roi = ['glasser_',num2str(glasser_indices(roi))];
    hcp_hdr = spm_vol(fullfile(cfg.dirs.derived_dir,'roi',['HCP-MMP1_on_MNI152_ICBM2009a_nlin.nii']));
    hcp_vol = spm_read_vols(hcp_hdr);
    
    evc_hdr = hcp_hdr;
    evc_hdr.fname = fullfile(glasser_dir,[this_roi,'_mask.nii']);
    evc_hdr.descrip = ['Glasser atlas with parcels ', num2str(glasser_indices(roi))];
    evc_hdr.dt = [2 0];
    evc_hdr.pinfo = [1 0 352]';
    evc_hdr.private.dat.dtype = 'FLOAT32-LE';
    evc_hdr.private.dat.scl_slope = 1;
    evc_hdr.private.dat.scl_inter = 0;
    evc_vol = ismember(round(hcp_vol),ROI);
    spm_write_vol(evc_hdr,evc_vol);
    
    clear matlabbatch
    % setup some specifics for the transformation
    struct_dir = fullfile(cfg.sub(i_sub).dir,'alldata','struct');
    normparams_path = spm_select('fplist',fullfile(struct_dir),['^iy_.*\.(nii|img)$']); %path to inverse transformation file
    matlabbatch{1}.spm.spatial.normalise.write.subj.def = {normparams_path};
    mask_path = fullfile(cfg.dirs.derived_dir,'roi','glasser_roi_masks',[this_roi,'_mask.nii']);
    matlabbatch{1}.spm.spatial.normalise.write.subj.resample = {mask_path};
    % get a functional hdr for defining the bounding box and the voxel size
    fnames = cellstr(spm_select('fplistrec',fullfile(cfg.sub(i_sub).dir,'alldata'),['^meanf.*\.(nii|img)$']));
    func_hdr = spm_vol(fnames{1});
    [bbox,vox] = spm_get_bbox(func_hdr,'fv');
    matlabbatch{1}.spm.spatial.normalise.write.woptions.bb = bbox;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.vox = vox;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.interp = 4;
    matlabbatch{1}.spm.spatial.normalise.write.woptions.prefix = 'indiv_';
    
    disp('Transforming current ROI mask from MNI into subject space')
    spm_jobman('run',matlabbatch)
    disp('Done.');
    
    %move file into subject directory
    movefile(fullfile(cfg.dirs.derived_dir,'roi','glasser_roi_masks',['indiv_',this_roi,'_mask.nii']),fullfile(cfg.sub(i_sub).dir,'roi'),'f');
    %rename file
    movefile(fullfile(cfg.sub(i_sub).dir,'roi',['indiv_',this_roi,'_mask.nii']),fullfile(cfg.sub(i_sub).dir,'roi',[this_roi,'_mask.nii']));
    
    clear matlabbatch
    matlabbatch{1}.spm.spatial.coreg.write.ref = {fnames{1}};
    matlabbatch{1}.spm.spatial.coreg.write.source = {fullfile(cfg.sub(i_sub).dir,'roi',[this_roi,'_mask.nii'])};
    matlabbatch{1}.spm.spatial.coreg.write.roptions.interp = 1;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.wrap = [0 0 0];
    matlabbatch{1}.spm.spatial.coreg.write.roptions.mask = 0;
    matlabbatch{1}.spm.spatial.coreg.write.roptions.prefix = 'r';
    
    disp('Resampling individual ROI mask to functional space')
    spm_jobman('run',matlabbatch)
    
    %% get the n most-activated voxels for each ROI
    
    % setup an array with Ns for writing ROIs
    nmost_array= [300];
    
    for nmost_idx = 1:length(nmost_array)
        nmost = nmost_array(nmost_idx);
        write_ROI_mask_all_bigger_baseline_nmost(cfg, i_sub,this_roi,nmost);
        write_ROI_mask_all_bigger_baseline_exp_nmost(cfg, i_sub,this_roi,nmost);
        
        movefile(fullfile(cfg.sub(i_sub).dir,'roi',[this_roi,'_mask_exp_nmost',num2str(nmost),'.nii']),fullfile(cfg.sub(i_sub).dir,'roi','glasser_masks',[this_roi,'_mask_exp_nmost',num2str(nmost),'.nii']));
        movefile(fullfile(cfg.sub(i_sub).dir,'roi',[this_roi,'_mask_loc_nmost',num2str(nmost),'.nii']),fullfile(cfg.sub(i_sub).dir,'roi','glasser_masks',[this_roi,'_mask_loc_nmost',num2str(nmost),'.nii']));
        
    end
    
    write_ROI_mask_full_anatomical(cfg, i_sub,this_roi);
    movefile(fullfile(cfg.sub(i_sub).dir,'roi',[this_roi,'_mask_full_anatomical.nii']),fullfile(cfg.sub(i_sub).dir,'roi','glasser_masks',[this_roi,'_mask_full_anatomical.nii']));
    
    
end



%% subfunctions

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
    end
end