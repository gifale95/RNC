% This function runs the normalization of some images for a specific
% subject i_sub.
% Inputs:
% i_sub = number of subject
% cfg = output of config_subject_rcor...
% smoothing = boolean indicating if you want to smooth as well or to only
% normalize
% sel_string = string to select the images, should be a path in combination
% with a regular expression e.g.
% "fullfile('/scratch/pablooyarzo/proj1/fMRI/derived_2/',['sub',num2str(i_sub,'%02i')],'/results/decoding/object_decoding/pairwise/'),'_v2.nii$')"
% 

function cfg = run_normalization(i_sub,cfg,smoothing,sel_string)

    
    FWHM = cfg.FWHM;
    
    %select files to normalize and smooth here
    fnames = cellstr(spm_select('fplistrec',sel_string));

    if ~isempty(fnames)
        
        loadtoolbox = 1 ;
        spm_jobman('initcfg')
        plot_sn(i_sub);
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
        
        if smoothing
        % smooth
        matlabbatch{2}.spm.spatial.smooth.data(1) = cfg_dep('Normalise: Write: Normalised Images (Subj 1)', substruct('.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}, '.','val', '{}',{1}), substruct('()',{1}, '.','files'));
        matlabbatch{2}.spm.spatial.smooth.fwhm = [FWHM FWHM FWHM];
        matlabbatch{2}.spm.spatial.smooth.dtype = 0 ;
        matlabbatch{2}.spm.spatial.smooth.im = 0 ;
        matlabbatch{2}.spm.spatial.smooth.prefix = sprintf('s%02d',FWHM);
        spm_jobman('run',matlabbatch)
        end
        
    end
end