%% script for aggregateting the searchlight decoding results 
clear all 
clc 
% set plot defaults 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3)   

%setup paths

path = '/data/pt_02348/objdraw/derived/';
fmri_path = '/data/pt_02350/derived';
save_path = '/data/pt_02348/objdraw/matlab/object_drawing_fusion/stats/searchlight_fusion_stats'; 

out_dir = '/data/pt_02348/objdraw/group_level/fmri/searchlight/';

% add tdt toolbox 

addpath(genpath('/data/pt_02348/objdraw/fmri/tdt_3.999'));

% add spm 

addpath('/data/pt_02348/objdraw/fmri/spm/spm12')

% add utils 
addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri/utils');
addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/meg/utils');
addpath(genpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/stats/'))

% get sub ids for all available subs

subs = dir(fullfile(path,'*od*'));
subs = {subs.name}';

% get fmri subnames 

fmri_subs = dir(fullfile(fmri_path,'*sub*'));
fmri_subs = {fmri_subs.name}';

% specify excluded subjects

fmri_excluded_subs = {'od07a','od09a','od10a','od13a','od22a','od23a','od29a'}; % bad in meg ? need to check -> 'od10a','od11a','od21a'

photo_RDVs = [];
drawing_RDVs = [];
sketch_RDVs = []; 

%specify results name
res_name = 'hrf_fitting_maskf16';

% load searchlight results

for sub_no = 1:length(subs)
    
    sub_id = subs{sub_no};
    
    if ~any(ismember(fmri_excluded_subs,sub_id))
        
        % load fMRI RDMs
        fmri_fname_photo = fullfile(fmri_path,['sub', sub_id(3:4)], 'results','decoding',res_name,'searchlight', 'Photo','s05res_accuracy_pairwise.nii');
        fmri_fname_drawing = fullfile(fmri_path,['sub', sub_id(3:4)], 'results','decoding',res_name,'searchlight', 'Drawing','s05res_accuracy_pairwise.nii');
        fmri_fname_sketch = fullfile(fmri_path,['sub', sub_id(3:4)], 'results','decoding',res_name,'searchlight', 'Sketch','s05res_accuracy_pairwise.nii');
        
        if exist(fmri_fname_sketch)
            fprintf('Loading fMRI %s\n',['sub',sub_id(3:4)]);
            
            %load(fmri_fname_photo);
            photo_RDVs = cat(2, photo_RDVs, cat(2,spm_read_vols(spm_vol(fmri_fname_photo))));
            %load(fmri_fname_drawing);
            drawing_RDVs = cat(2, drawing_RDVs, cat(2,spm_read_vols(spm_vol(fmri_fname_drawing))));
            %load(fmri_fname_sketch);
            sketch_RDVs = cat(2, sketch_RDVs, cat(2,spm_read_vols(spm_vol(fmri_fname_sketch))));
            
        else
            fprintf('Results not complete for sub %s\n',['sub',sub_id(3:4)]);
            
        end
        
    end
end

%% setup results volumes 

backgroundvalue = 0; 
hdr = spm_vol(fullfile('/data/pt_02350/derived/sub01/alldata/run01/', 'warf01-01-001.img')); %  this should be an image with the same dimensionality as the searchlight results
n_subs = size(photo_RDVs,2);
photo_resultsvol = backgroundvalue * ones([n_subs,hdr.dim(1:3)]); % prepare results volume with background value (default: 0)
drawing_resultsvol = backgroundvalue * ones([n_subs,hdr.dim(1:3)]);
sketch_resultsvol = backgroundvalue * ones([n_subs,hdr.dim(1:3)]);

for sub = 1:n_subs
photo_resultsvol(sub,results.mask_index) = photo_RDVs(:,sub);
drawing_resultsvol(sub,results.mask_index) = drawing_RDVs(:,sub);
sketch_resultsvol(sub,results.mask_index) = sketch_RDVs(:,sub);

end 
%% compute stats 

nperm = 10000;
cluster_th = 0.001;
significance_th = 0.05;
tail = 'right';

sig_photo = permutation_cluster_1sample_weight_alld (photo_resultsvol-50, nperm, cluster_th, significance_th, tail);
sig_drawing =  permutation_cluster_1sample_weight_alld (drawing_resultsvol-50, nperm, cluster_th, significance_th, tail);
sig_sketch =  permutation_cluster_1sample_weight_alld (sketch_resultsvol-50, nperm, cluster_th, significance_th, tail);

%% setup stats volumes and write them 

backgroundvalue = 0; 
hdr = spm_vol(fullfile('/data/pt_02350/derived/sub01/alldata/run01/', 'warf01-01-001.img')); %  this should be an image with the same dimensionality as the searchlight results

hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
hdr.descrip = sprintf('Searchlight between condition similarity stats');
hdr = rmfield(hdr, 'n');
hdr.fname = fullfile(out_dir,'photo_drawing_sim_searchlight_stats.nii');
spm_write_vol(hdr, sig_photo_drawing);
hdr.fname = fullfile(out_dir,'photo_sketch_sim_searchlight_stats.nii');
spm_write_vol(hdr, sig_photo_sketch);
hdr.fname = fullfile(out_dir, 'drawing_sketch_sim_searchlight_stats.nii');
spm_write_vol(hdr, sig_drawing_sketch);

%% mask results volume with stats and write mean images 


hdr = spm_vol(fullfile('/data/pt_02350/derived/sub01/alldata/run01/', 'warf01-01-001.img')); %  this should be an image with the same dimensionality as the searchlight results
hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
hdr.descrip = sprintf('Searchlight decoding');
hdr = rmfield(hdr, 'n');
hdr.fname = fullfile(out_dir,'photo_decoding_searchlight_masked.nii');
mean_photo = squeeze(mean(photo_resultsvol));
spm_write_vol(hdr, mean_photo);
hdr.fname = fullfile(out_dir,'drawing_decoding_searchlight_masked.nii');
mean_drawing = squeeze(mean(drawing_resultsvol));
spm_write_vol(hdr, mean_drawing);
hdr.fname = fullfile(out_dir, 'sketch_decoding_searchlight_masked.nii');
mean_sketch = squeeze(mean(sketch_resultsvol));
spm_write_vol(hdr, mean_sketch);

%% compute sum of overlap between significance masks 

overlap_photo_drawing = sum(sig_photo.*sig_drawing,'all')/sum(sig_photo,'all');
overlap_photo_sketch = sum(sig_photo.*sig_sketch,'all')/sum(sig_photo,'all');
overlap_drawing_sketch = sum(sig_drawing.*sig_sketch,'all')/sum(sig_sketch,'all');

%% compute conjunction for all depictions 

conjunction_map = sig_photo.*sig_drawing.*sig_sketch; 

hdr.fname = fullfile(out_dir, 'conjunction_map.nii');
spm_write_vol(hdr, conjunction_map);

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

plot_data_on_axial(fullfile(out_dir, 'photo_decoding_searchlight_masked.nii'),fullfile(out_dir, 'photo_decoding_searchlight_masked.nii'), [],cmap)
export_fig(fullfile(results_path,'searchlight_photo_decoding_map.png'), '-png', '-r300','-transparent','-nocrop'); % -r300 is the PPI value, default resolution is low

