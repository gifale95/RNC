%% script for correlating the searchlight RDMs for different conditions 

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

%% load searchlight results 

for sub_no = 1:length(subs)
    
    sub_id = subs{sub_no};
    
    if ~any(ismember(fmri_excluded_subs,sub_id))
    
    fprintf('Loading fMRI %s\n',['sub',sub_id(3:4)]);
   
    % load fMRI RDMs 
    fmri_fname_photo = fullfile(fmri_path,['sub', sub_id(3:4)], 'results','RSA_noisenorm_pearson_samevoxsz','searchlight', 'Photo','res_other_average_RDV.mat');
    fmri_fname_drawing = fullfile(fmri_path,['sub', sub_id(3:4)], 'results','RSA_noisenorm_pearson_samevoxsz','searchlight', 'Drawing','res_other_average_RDV.mat');
    fmri_fname_sketch = fullfile(fmri_path,['sub', sub_id(3:4)], 'results','RSA_noisenorm_pearson_samevoxsz','searchlight', 'Sketch','res_other_average_RDV.mat');
    
    load(fmri_fname_photo);
    photo_RDVs = cat(3, photo_RDVs, cat(2,results.other_average_RDV.output{:}));
    load(fmri_fname_drawing);
    drawing_RDVs = cat(3, drawing_RDVs, cat(2,results.other_average_RDV.output{:}));
    load(fmri_fname_sketch);
    sketch_RDVs = cat(3, sketch_RDVs, cat(2,results.other_average_RDV.output{:}));
    end 
end

%% compute correlation between all searchlight RDVs for two conditions for every subject seperately

for sub = 1:size(photo_RDVs,3)
    
    photo_drawing_sim(:,sub) = arrayfun(@(col) corr(1-photo_RDVs(:,col,sub), 1-drawing_RDVs(:,col,sub),'type','pearson'), 1:size(photo_RDVs,2), 'Uni', 1);
    photo_sketch_sim(:,sub) = arrayfun(@(col) corr(1-photo_RDVs(:,col,sub), 1-sketch_RDVs(:,col,sub),'type','pearson'), 1:size(photo_RDVs,2), 'Uni', 1);
    drawing_sketch_sim(:,sub) = arrayfun(@(col) corr(1-drawing_RDVs(:,col,sub), 1-sketch_RDVs(:,col,sub),'type','pearson'), 1:size(photo_RDVs,2), 'Uni', 1);
    
end 

%% setup results volumes and write them 

backgroundvalue = 0; 
hdr = spm_vol(fullfile('/data/pt_02350/derived/sub01/alldata/run01/', 'warf01-01-001.img')); %  this should be an image with the same dimensionality as the searchlight results
n_subs = size(photo_RDVs,3);
photo_drawing_resultsvol = backgroundvalue * ones([n_subs,hdr.dim(1:3)]); % prepare results volume with background value (default: 0)
photo_sketch_resultsvol = backgroundvalue * ones([n_subs,hdr.dim(1:3)]);
drawing_sketch_resultsvol = backgroundvalue * ones([n_subs,hdr.dim(1:3)]);

for sub = 1:n_subs
photo_drawing_resultsvol(sub,results.mask_index) = photo_drawing_sim(:,sub);
photo_sketch_resultsvol(sub,results.mask_index) = photo_sketch_sim(:,sub);
drawing_sketch_resultsvol(sub,results.mask_index) = drawing_sketch_sim(:,sub);

end 

% hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
% hdr.descrip = sprintf('Seearchlight between condition similarity');
% hdr = rmfield(hdr, 'n');
% hdr.fname = fullfile(out_dir,'photo_drawing_sim_searchlight.nii');
% spm_write_vol(hdr, mean(photo_drawing_resultsvol));
% hdr.fname = fullfile(out_dir,'photo_sketch_sim_searchlight.nii');
% spm_write_vol(hdr, mean(photo_sketch_resultsvol));
% hdr.fname = fullfile(out_dir, 'drawing_sketch_sim_searchlight.nii');
% spm_write_vol(hdr, mean(drawing_sketch_resultsvol));

%% compute stats 

nperm = 10000;
cluster_th = 0.001;
significance_th = 0.05;
tail = 'right';

sig_photo_drawing = permutation_cluster_1sample_weight_alld (photo_drawing_resultsvol, nperm, cluster_th, significance_th, tail);
sig_photo_sketch =  permutation_cluster_1sample_weight_alld (photo_sketch_resultsvol, nperm, cluster_th, significance_th, tail);
sig_drawing_sketch =  permutation_cluster_1sample_weight_alld (drawing_sketch_resultsvol, nperm, cluster_th, significance_th, tail);

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

%% plot results

addpath('/data/pt_02348/objdraw/matlab/object_drawing_fusion/fmri/utils/searchlight')

fig = plot_data_on_axial(fullfile(out_dir,'photo_drawing_sim_searchlight.nii'),fullfile(out_dir,'photo_drawing_sim_searchlight_stats.nii'),'Photo-Drawing Similarity (p>0.05)');
imwrite(fig.cdata,fullfile(out_dir,'searchlight_photo_drawing_sim_sig_p_.05.png'),'jpg'); 

fig = plot_data_on_axial(fullfile(out_dir,'photo_sketch_sim_searchlight.nii'),fullfile(out_dir,'photo_sketch_sim_searchlight_stats.nii'),'Photo-Sketch Similarity (p>0.05)');
imwrite(fig.cdata,fullfile(out_dir,'searchlight_photo_sketch_sim_sig_p_.05.png'),'png'); 

fig = plot_data_on_axial(fullfile(out_dir,'drawing_sketch_sim_searchlight.nii'),fullfile(out_dir,'drawing_sketch_sim_searchlight_stats.nii'),'Drawing-Sketch Similarity (p<0.05)');
imwrite(fig.cdata,fullfile(out_dir,'searchlight_drawing_sketch_sim_sig_p_.05.png'),'png'); 
