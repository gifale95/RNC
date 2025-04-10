%% script for inspecting bad volumes

clear all
clc

addpath('/data/pt_02348/objdraw/snips/martin_spm')

% get config for experiment

cfg = config_subjects_objdraw();

for i_sub = 30%:length(cfg.sub_indices)
    
    sub_dir = fullfile(cfg.dirs.derived_dir,sprintf('sub%02d',i_sub));
    prefix =  cfg.prefix;
    
    P = []; % all headers
    ct = []; % counter of where a run ends
    
    % Functional runs
    for i_run = cfg.sub(i_sub).run_indices
        fname_path = fullfile(sub_dir,'alldata',sprintf('run%02d',i_run));
        fname = spm_select('fplist',fname_path,['^' prefix '.*\.img$']);
        P = [P; spm_vol(fname)]; % get headers
        ct = [ct length(P)];
    end
    % Localizer runs
    for i_run = 1:length(cfg.sub(i_sub).import.localizer)
        fname_path = fullfile(sub_dir,'alldata','localizer');
        fname = spm_select('fplist',fname_path,['^' prefix '.*\.img$']);
        P = [P; spm_vol(fname)]; % get headers
        ct = [ct length(P)];
    end
    
    % Selecting DICOMs (numerical format!)
    
    prefix = 'f';
    maskprefix = ['emean' prefix cfg.prefix];
    % if maskname is not included, the eyes - which cause large amounts of
    % variance - are still in, obscuring outlier volumes
    maskname = spm_select('fplist',fullfile(cfg.sub(i_sub).dir,'alldata','run01'),['^' maskprefix '.*\.img$']);
    check_data = 0;
    include_loc = 1;
    plot_on = 1 ;
    [outlier_im, outlier_slices] = check_slices(cfg,i_sub,prefix,include_loc,check_data,plot_on,maskname);
    
    for im = 1:length(outlier_im)
        
        hdr = P(outlier_im(im));
        vol = spm_read_vols(hdr);
        disp(hdr.dim)
        figure
        imagesc(transform_vol(vol));
        colormap('gray')
    end
end