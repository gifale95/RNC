%% get the brain coverage of a single subject for visualization 
function get_brain_coverage(cfg,i_sub)

% fill resultsvol 4D and write 4D nifi
backgroundvalue = NaN;
% get canonical hdr from first preprocesed functional file
template_file = dir(fullfile(cfg.sub(i_sub).dir, 'alldata','run01','*.nii'));
template_file = fullfile(template_file(1).folder,template_file(1).name);
hdr= spm_vol(template_file); % choose canonical hdr from first classification image
hdr = rmfield(hdr,'pinfo');
%hdr = rmfield(hdr, 'dt');

resultsvol_hdr = hdr;
resultsvol_hdr.fname = fullfile(cfg.sub(i_sub).dir,'alldata','other', ['brain_coverage.nii']);
resultsvol_hdr.descrip = sprintf('Brain Coverage Map');
resultsvol = backgroundvalue * ones(resultsvol_hdr.dim(1:3)); % prepare results volume with background value (default: 0)
spm_write_vol(resultsvol_hdr,resultsvol);
end 
