%% evaluate decision values 
clear all

sub_id = 'sub03'; 

outdir = ['/Users/johannessinger/scratch/dfg_projekt/WP1/derived/pilot/',sub_id, '/results/decoding/crossdecoding_only_controversial_hrf_fitting/searchlight'];
roi_dir = ['/Users/johannessinger/scratch/dfg_projekt/WP1/derived/pilot/',sub_id, '/roi']; 
load(fullfile(outdir,'res_decision_values.mat'));

manmade_manmade_vals = zeros(1,length(results.decision_values.output)); 
manmade_natural_vals = zeros(1,length(results.decision_values.output)); 
natural_manmade_vals = zeros(1,length(results.decision_values.output)); 
natural_natural_vals = zeros(1,length(results.decision_values.output)); 


for i = 1:length(results.decision_values.output)
    
    these_dec_vals = results.decision_values.output{i};
    
    manmade_natural_vals(i) = mean(these_dec_vals(1:15)); 
    natural_manmade_vals(i) = mean(these_dec_vals(16:30));
end 

%% write searchlight maps 
% fill resultsvol 4D and write 4D nifi
backgroundvalue = NaN;
% get canonical hdr from first preprocesed functional file
template_file = dir(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived/pilot/',sub_id, '/alldata/run01','*.nii'));
template_file = fullfile(template_file(1).folder,template_file(1).name);
hdr= spm_vol(template_file); % choose canonical hdr from first classification image
hdr = rmfield(hdr,'pinfo');
%hdr = rmfield(hdr, 'dt');

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

resultsvol_hdr = hdr;
resultsvol_hdr.fname = fullfile(outdir,'incongruent_controversial_difference.nii');
resultsvol_hdr.descrip = sprintf('Difference decision value map');
resultsvol = backgroundvalue * ones(resultsvol_hdr.dim(1:3)); % prepare results volume with background value (default: 0)
resultsvol(results.mask_index) = abs(manmade_natural_vals-natural_manmade_vals);
spm_write_vol(resultsvol_hdr,resultsvol);


%% evaluate results based on ROIs 

evc_mask = spm_read_vols(spm_vol(fullfile(roi_dir, 'evcmask.nii'))); 
IT_mask = spm_read_vols(spm_vol(fullfile(roi_dir, 'PPA_mask.nii')));

evc_mask = evc_mask(results.mask_index); 
IT_mask = IT_mask(results.mask_index); 

evc_manmade_natural = manmade_natural_vals(logical(evc_mask)); 
evc_natural_manmade = natural_manmade_vals(logical(evc_mask));
evc_natural_natural = natural_natural_vals(logical(evc_mask));
evc_manmade_manmade = manmade_manmade_vals(logical(evc_mask));

IT_manmade_natural = manmade_natural_vals(logical(IT_mask)); 
IT_natural_manmade = natural_manmade_vals(logical(IT_mask));
IT_natural_natural = natural_natural_vals(logical(IT_mask));
IT_manmade_manmade = manmade_manmade_vals(logical(IT_mask));

% plot 

figure 

mean_evc_vals = [mean(evc_manmade_manmade) mean(evc_manmade_natural) mean(evc_natural_manmade) mean(evc_natural_natural)]; 
mean_IT_vals = [mean(IT_manmade_manmade) mean(IT_manmade_natural) mean(IT_natural_manmade) mean(IT_natural_natural)]; 

all_mean_vals = cat(2, mean_evc_vals', mean_IT_vals')


bar(all_mean_vals);
ylim([-0.3 0.3]); 
legend({'EVC', 'IT'} ,'Location','southeast')
xticklabels({'EVC-manmade IT-manmade'; 'EVC-manmade IT-natural'; 'EVC-natural IT-manmade';'EVC-natural IT-natural'})


