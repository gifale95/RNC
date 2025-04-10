%% format searchlight fusion results 
% this function can be used to load the searchlight fusion results of one
% subject and to convert the results to a 4D nifi file and write that file 

function format_fusion_results(cfg,outdir,cfgd,i_sub,cond)

fprintf('Formatting %s searchlight fusion results for subject %s \n', cond, num2str(i_sub)); 

file = dir(fullfile(outdir, '*res_other_average_RDV.mat'));
filename = {file.name}';
load(fullfile(outdir,filename{:}))

output = cat(2,results.other_average_RDV.output{:});

% fill resultsvol 4D and write 4D nifi
backgroundvalue = 0;
% get canonical hdr from first preprocesed functional file 
template_file = dir(fullfile(cfgd.sub(i_sub).dir, 'alldata','run01','*.img'));
template_file = fullfile(template_file(1).folder,template_file(1).name);
hdr= read_header('SPM12',template_file); % choose canonical hdr from first classification image
hdr = rmfield(hdr,'pinfo');
%hdr = rmfield(hdr, 'dt');

% compute fusion all in one step 
for time=1:size(cfg.meg_RDM,3)
meg_RDV(:,time) = squareformq(cfg.meg_RDM(:,:,time)');
end 
resultsvol_4d = corr(output,meg_RDV,'Type','Spearman');

%write single niftis
for time = 1:size(cfg.meg_RDM,3)
    
    resultsvol_hdr = hdr;
    resultsvol_hdr.fname = fullfile(outdir,sprintf('searchlight_fusion_4d_time_%i.nii',time));
    resultsvol_hdr.descrip = sprintf('3D searchlight fusion map');
    resultsvol = backgroundvalue * ones(resultsvol_hdr.dim(1:3)); % prepare results volume with background value (default: 0)
    resultsvol(results.mask_index) = resultsvol_4d(:,time); 
    spm_write_vol(resultsvol_hdr,resultsvol);
end

%merge single 3d nifitis to 4d nifti
nii_in = dir(fullfile(outdir,'*time*.nii'));
if length(nii_in)>1
    fprintf('...compressing 3-D NIFTI files into 4-D NIFTI file...\n')
    nii_file = {};
    nii_out  = fullfile(outdir,'searchlight_fusion_4d.nii');
    
    for b=1:length(nii_in)
        nii_file{b} = fullfile(outdir,nii_in(b).name);
    end
    
    nii_files = natsort(nii_file); 
    V = spm_vol(strvcat(nii_file{:}));
    
    spm_file_merge(V, nii_out); 
    %nii_3Dto4D(nii_file,nii_out);
    
    for b=1:length(nii_file)
        delete(nii_file{b});
    end
end
end
