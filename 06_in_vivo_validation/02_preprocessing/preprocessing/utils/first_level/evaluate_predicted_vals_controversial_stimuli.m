%% evaluate decision values 

outdir = '/Users/johannessinger/scratch/dfg_projekt/WP1/derived/sub01/results/decoding/crossdecoding_controversial/searchlight';
manmade_manmade_vals = zeros(1,length(results.predicted_labels.output)); 
manmade_natural_vals = zeros(1,length(results.predicted_labels.output)); 
natural_manmade_vals = zeros(1,length(results.predicted_labels.output)); 
natural_natural_vals = zeros(1,length(results.predicted_labels.output)); 


for i = 1:length(results.predicted_labels.output)
    
    these_dec_vals = results.predicted_labels.output{i};
    
    manmade_manmade_vals(i) = mean(these_dec_vals(1:15)); 
    manmade_natural_vals(i) = mean(these_dec_vals(16:30)); 
    natural_manmade_vals(i) = mean(these_dec_vals(31:45));
    natural_natural_vals(i) = mean(these_dec_vals(46:end));
    
end 


%% write searchlight maps 

% fill resultsvol 4D and write 4D nifi
backgroundvalue = NaN;
% get canonical hdr from first preprocesed functional file
template_file = dir(fullfile('/Users/johannessinger/Documents/cloud_Berlin/Projekte/dfg/WP1/derived/sub01/alldata/run01','*.nii'));
template_file = fullfile(template_file(1).folder,template_file(1).name);
hdr= spm_vol(template_file); % choose canonical hdr from first classification image
hdr = rmfield(hdr,'pinfo');
%hdr = rmfield(hdr, 'dt');

resultsvol_hdr = hdr;
resultsvol_hdr.fname = fullfile(outdir,'low_level_manmade_high_level_manmade.nii');
resultsvol_hdr.descrip = sprintf('Mean decision value map');
resultsvol = backgroundvalue * ones(resultsvol_hdr.dim(1:3)); % prepare results volume with background value (default: 0)
resultsvol(results.mask_index) = manmade_manmade_vals;
spm_write_vol(resultsvol_hdr,resultsvol);

resultsvol_hdr = hdr;
resultsvol_hdr.fname = fullfile(outdir,'low_level_natural_high_level_natural.nii');
resultsvol_hdr.descrip = sprintf('Mean decision value map');
resultsvol = backgroundvalue * ones(resultsvol_hdr.dim(1:3)); % prepare results volume with background value (default: 0)
resultsvol(results.mask_index) = natural_natural_vals;
spm_write_vol(resultsvol_hdr,resultsvol);

resultsvol_hdr = hdr;
resultsvol_hdr.fname = fullfile(outdir,'low_level_manmade_high_level_natural.nii');
resultsvol_hdr.descrip = sprintf('Mean decision value map');
resultsvol = backgroundvalue * ones(resultsvol_hdr.dim(1:3)); % prepare results volume with background value (default: 0)
resultsvol(results.mask_index) = manmade_natural_vals;
spm_write_vol(resultsvol_hdr,resultsvol);

resultsvol_hdr = hdr;
resultsvol_hdr.fname = fullfile(outdir,'low_level_natural_high_level_manmade.nii');
resultsvol_hdr.descrip = sprintf('Mean decision value map');
resultsvol = backgroundvalue * ones(resultsvol_hdr.dim(1:3)); % prepare results volume with background value (default: 0)
resultsvol(results.mask_index) = natural_manmade_vals;
spm_write_vol(resultsvol_hdr,resultsvol);

%% restrict the analysis to predefined ROIs

