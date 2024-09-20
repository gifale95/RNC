%% script for aggregateting the searchlight decoding results 
clear all 
clc 
% set plot defaults 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3)   

%setup paths
if ismac
fmri_path = '/Users/johannessinger/scratch/dfg_projekt/WP1/derived/pilot/';

out_dir = '/Users/johannessinger/scratch/dfg_projekt/WP1/derived/pilot/group';

elseif isunix 
    
end 

% add stats functions 
addpath(genpath('/Users/johannessinger/scratch/dfg_projekt/WP1/analysis/stats'))

% % add tdt toolbox 
% 
% addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'));
% 
% % add spm 
% 
% addpath('/data/pt_02348/objdraw/fmri/spm/spm12')
% 
% % add utils 
% addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri/utils/searchlight');
% addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/meg/utils');
% addpath(genpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/stats/'))

% get fmri subnames 

fmri_subs = dir(fullfile(fmri_path,'*sub*'));
fmri_subs = {fmri_subs.name}';

% specify excluded subjects

fmri_excluded_subs = {'od07a','od09a','od10a','od13a','od22a','od23a','od29a'}; % bad in meg ? need to check -> 'od10a','od11a','od21a'

decoding_maps = [];


%specify results name
res_name = 'decoding_nobetas_controversial_type';
fname = 's05wres_accuracy_minus_chance.nii'; %'s05wres_accuracy_minus_chance.nii' ;

% load searchlight results

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
    
    if ~any(ismember(fmri_excluded_subs,sub_id))
        
        % load fMRI RDMs
        fmri_fname = fullfile(fmri_path,sub_id, 'results','decoding',res_name,'searchlight',fname);

        if exist(fmri_fname)
            fprintf('Loading fMRI %s\n',sub_id);
            
            decoding_maps = cat(4, decoding_maps, spm_read_vols(spm_vol(fmri_fname)));
        else
            fprintf('Results not complete for sub %s\n',sub_id);
            
        end
        
    end
end

%% compute stats 

nperm = 1000;
cluster_th = 0.001;
significance_th = 0.05;
tail = 'right';

sig_searchlight = permutation_cluster_1sample_weight_alld (permute(decoding_maps,[4 1 2 3]), nperm, cluster_th, significance_th, tail);

%% write mean results without stats masking

hdr = spm_vol(fmri_fname); %  this should be an image with the same dimensionality as the searchlight results
hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
hdr.descrip = sprintf('Searchlight decoding');
hdr = rmfield(hdr, 'n');
hdr.fname = fullfile(out_dir,[res_name,'_accuracy_minus_chance.nii']); %accuracy_minus_chance
mean_vol = squeeze(mean(decoding_maps,4));
spm_write_vol(hdr, mean_vol);

% %% write coverage image 
% hdr = spm_vol(fmri_fname); %  this should be an image with the same dimensionality as the searchlight results
% hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
% hdr.descrip = sprintf('Brain_Coverage');
% hdr = rmfield(hdr, 'n');
% hdr.fname = fullfile(out_dir,['brain_coverage.nii']); %accuracy_minus_chance
% mean_vol = squeeze(mean(decoding_maps,4));
% mean_vol(mean_vol ~=0) = 1; 
% spm_write_vol(hdr, mean_vol);

%% mask results volume with stats and write mean images 


hdr = spm_vol(fmri_fname_photo); %  this should be an image with the same dimensionality as the searchlight results
hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
hdr.descrip = sprintf('Searchlight decoding masked');
hdr = rmfield(hdr, 'n');
hdr.fname = fullfile(out_dir,[res_name,'_accuracy_minus_chance_masked.nii']);
spm_write_vol(hdr, mean_vol.*sig_searchlight);

%% setup stats volumes and write them 


backgroundvalue = 0; 
hdr = spm_vol(fmri_fname_photo); %  this should be an image with the same dimensionality as the searchlight results

hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
hdr.descrip = sprintf('Searchlight decoding difference');
hdr = rmfield(hdr, 'n');
hdr.fname = fullfile(out_dir,'photo_decoding_searchlight_stats_maskf16.nii');
spm_write_vol(hdr, sig_searchlight_photo);
hdr.fname = fullfile(out_dir,'drawing_decoding_searchlight_stats_maskf16.nii');
spm_write_vol(hdr, sig_searchlight_drawing);
hdr.fname = fullfile(out_dir, 'sketch_decoding_searchlight_stats_maskf16.nii');
spm_write_vol(hdr, sig_searchlight_sketch);

backgroundvalue = 0; 
hdr = spm_vol(fmri_fname_photo); %  this should be an image with the same dimensionality as the searchlight results

hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
hdr.descrip = sprintf('Searchlight decoding difference');
hdr = rmfield(hdr, 'n');
hdr.fname = fullfile(out_dir,'photo_drawing_decoding_searchlight_stats_maskf16.nii');
spm_write_vol(hdr, sig_searchlight_photo_drawing);
hdr.fname = fullfile(out_dir,'photo_sketch_decoding_searchlight_stats_maskf16.nii');
spm_write_vol(hdr, sig_searchlight_photo_sketch);
hdr.fname = fullfile(out_dir, 'drawing_sketch_decoding_searchlight_stats_maskf16.nii');
spm_write_vol(hdr, sig_searchlight_drawing_sketch);

%% compute sum of overlap between significance masks 

overlap_photo_drawing = sum(sig_searchlight_photo.*sig_searchlight_drawing,'all')/sum(sig_searchlight_photo,'all');
overlap_photo_sketch = sum(sig_searchlight_photo.*sig_searchlight_sketch,'all')/sum(sig_searchlight_photo,'all');
overlap_drawing_sketch = sum(sig_searchlight_drawing.*sig_searchlight_sketch,'all')/sum(sig_searchlight_sketch,'all');

%% compute conjunction for all depictions - all possible combinations for coloring 

photo_and_drawing_and_sketch = sig_searchlight_photo&sig_searchlight_drawing&sig_searchlight_sketch;
photo_and_drawing = sig_searchlight_photo & sig_searchlight_drawing & ~sig_searchlight_sketch; 
photo_and_sketch =sig_searchlight_photo & ~sig_searchlight_drawing & sig_searchlight_sketch; 
drawing_and_sketch = ~sig_searchlight_photo & sig_searchlight_drawing & sig_searchlight_sketch; 
only_photo = sig_searchlight_photo & ~sig_searchlight_drawing & ~sig_searchlight_sketch; 
only_drawing = ~sig_searchlight_photo & sig_searchlight_drawing & ~sig_searchlight_sketch; 
only_sketch = ~sig_searchlight_photo & ~sig_searchlight_drawing & sig_searchlight_sketch; 

%hdr.fname = fullfile(out_dir, 'conjunction_map.nii');
%spm_write_vol(hdr, conjunction_map);

%% load conjuction map if not loaded 

%set results path 
results_path = '/data/pt_02348/objdraw/group_level/fmri/searchlight/';

%load colormap
cmap = colormap('redblueTecplot');
close all

%add plotting functions 
addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri/utils/searchlight')

[~,fh] = plot_data_on_axial(fullfile(out_dir, 'conjunction_map.nii'),fullfile(out_dir, 'conjunction_map.nii'), [],cmap)
set(fh,'Color','none')
export_fig(fullfile(results_path,'conjunction_map.png'), '-png', '-r300','-transparent','-nocrop'); % -r300 is the PPI value, default resolution is low

%% plot searchlight results 

plot_data_on_axial(fullfile(out_dir, 'photo_decoding_searchlight_masked_maskf16.nii'),fullfile(out_dir, 'photo_decoding_searchlight_stats_maskf16.nii'), [],cmap)
export_fig(fullfile(results_path,'searchlight_photo_decoding_map_maskf16.png'), '-png', '-r300','-transparent','-nocrop'); % -r300 is the PPI value, default resolution is low

%% plot full conjunction 

%set results path 
results_path = '/data/pt_02348/objdraw/group_level/fmri/searchlight/';

% load a hdr
hdr = spm_vol(fmri_fname_photo); %  this should be an image with the same dimensionality as the searchlight results

%load colormap
cmap = colormap('redblueTecplot');
close all

plot_conjunction_additive_coloring(hdr, photo_and_drawing_and_sketch, photo_and_drawing, photo_and_sketch, drawing_and_sketch, sig_searchlight_photo, sig_searchlight_drawing, sig_searchlight_sketch, '',cmap)
%export_fig(fullfile(results_path,'conjunction_map_with_overlaps_otherT1.svg'), '-svg', '-r600','-transparent'); % -r300 is the PPI value, default resolution is low

print(fullfile(results_path,'conjunction_map_with_overlaps_maskf16.svg'), ...
              '-dsvg', '-r600')

