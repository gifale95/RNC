function standard_glmdenoise(cfg,i_sub,resname,prefix,outname)

boot_on = 0;
overwritefigures = 1;

% load SPM
load(fullfile(cfg.sub(i_sub).dir,'results','GLM',resname,'SPM.mat'))

full_design = SPM.xX.X;

names = SPM.xX.name;

vol_ind = 0 ;
design = cell(1,length(SPM.Sess));

for i_run = 1:length(SPM.Sess)
    regressor_ind = find(~cellfun(@isempty,strfind(names,sprintf('Sn(%i)',i_run))));
    regressor_ind = regressor_ind(1:length(SPM.Sess(i_run).U));
    
    vol_ind = (vol_ind(end)+1):(vol_ind(end)+SPM.nscan(i_run));
    
    design{i_run} = full_design(vol_ind,regressor_ind);
    
    % check orthogonality and throw error otherwise
    all0 = sum(~any(design{i_run},1));
    if (rank(design{i_run})+all0)<size(design{i_run},2)
        error('Rank deficient model. Some regressors are a linear combination of others. Please check!')
    end
    
end

hdr = cell(1,cfg.sub(i_sub).n_runs);
vol = cell(size(hdr));

sub_dir = cfg.sub(i_sub).dir;

disp('Loading data...')
ct = 0 ;
for i_run = cfg.sub(i_sub).run_indices
    
    ct = ct+1 ;
    fprintf('run %i/%i\n',i_run,cfg.sub(i_sub).n_runs)
    % get file names
    run_dir = fullfile(sub_dir,'alldata',sprintf('run%02d',i_run));
    files = spm_select('FPList',run_dir,['^' prefix '.*\.img$']);
    
    disp('Reading run headers...')
    hdr{ct} = spm_vol(files);
    
    disp('Reading volumes')
    vol{ct} = single(zeros([hdr{ct}(1).dim(1:3) length(hdr{ct})]));
    for i_vol = 1:length(hdr{ct})
        fprintf('.')
        vol{ct}(:,:,:,i_vol) = single(spm_read_vols(hdr{ct}(i_vol)));
        if ~mod(i_vol,100)
            fprintf('\n')
        end
    end
    fprintf('\n')
end
disp('done.')

clear hdr

if boot_on
    boot = [];
else
    boot = struct('numboots',0,'numpcstotry',20,'overwritefigures',overwritefigures);
end

figures_dir = fullfile(cfg.dirs.home_dir,'figures',sprintf('sub%02i',i_sub),resname);
tic
% second output would be denoised data
results = GLMdenoisedata(design,vol,1,cfg.TR,'assume',1,boot,figures_dir);
toc

ct = 0 ;
for i_run = cfg.sub(i_sub).run_indices
    
    ct = ct+1 ;
    R = double(results.pcregressors{ct}(:,1:results.pcnum)); %#ok<NASGU>
    save(fullfile(cfg.sub(i_sub).dir,'alldata','parameters',sprintf('glmdenoise_%s_run%02i.mat',outname,i_run)),'R')
    
end