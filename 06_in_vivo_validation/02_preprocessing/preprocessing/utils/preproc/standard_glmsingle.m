function standard_glmsingle(cfg,i_sub,prefix,design,conditions,outdir)

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


opt = struct('wantmemoryoutputs',[0 0 0 1]);
% second output would be denoised data
results = GLMestimatesingletrial(design,vol,0.5,cfg.TR,outdir,opt);

% consolidate design matrices
designALL = cat(1,design{:});

% compute a vector containing 1-indexed condition numbers in chronological 
% order.

betas = results{1,4};
betas = betas.modelmd;

corder = [];
for p=1:size(designALL,1)
    if any(designALL(p,:))
        corder = [corder find(designALL(p,:))];
    end
end

example_hdr = hdr{1,1}(1);
example_hdr.descrip = 'Single-trial beta estimates';

% write beta images which can then be used for decoding 
conds = {'Photo';'Drawing';'Sketch'};
% setup stuff that we need for decoding 
chunks = zeros(1,length(corder)); 
labels = zeros(1,length(corder)); 
cond_ind = zeros(1,length(corder)); 

mkdir(fullfile(outdir,'betas')); 

trl_ct = 1; 
for run = 1:length(conditions) 
    
for idx = trl_ct:trl_ct+95
    example_hdr.fname = fullfile(outdir,'betas',[conds{conditions(run)},'_object_',num2str(corder(idx),'%02.f'),'_run_',num2str(run,'%02.f'), '_trial_', num2str(idx,'%04.f'),'.nii']);
    spm_write_vol(example_hdr, betas(:,:,:,idx)); 
    
    chunks(idx)= run; 
    labels(idx) = corder(idx);
    cond_ind(idx) = conditions(run); 
    
end 
trl_ct = idx+1; 
end 

% save the chunks labels and conditions for decoding 
save(fullfile(outdir,'chunks.mat'),'chunks')
save(fullfile(outdir,'labels.mat'),'labels')
save(fullfile(outdir,'conditions.mat'),'cond_ind')

end 
