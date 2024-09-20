%% script for aggregateting the searchlight decoding results 
clear all 
clc 
% set plot defaults 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3)   

%setup paths
if ismac
fmri_path = '/Users/johannessinger/scratch/rcor_collab/derived/';

out_dir = '/Users/johannessinger/scratch/rcor_collab/derived/group';

elseif isunix 
    
   fmri_path = '/scratch/singej96/rcor_collab/derived/';

out_dir = '/scratch/singej96/rcor_collab/derived/group'; 
end 

% add stats functions 
addpath(genpath('/scratch/singej96/rcor_collab/analysis/stats'))

% get config 
addpath('/scratch/singej96/rcor_collab/analysis/')
cfg = config_subjects_rcor_alldata;

% add spm
addpath(cfg.dirs.spm_dir); %#ok<*MCAP>
 

fmri_subs = dir(fullfile(fmri_path,'*sub*'));
fmri_subs = {fmri_subs.name}';

% specify excluded subjects

fmri_excluded_subs = {}; % bad in meg ? need to check -> 'od10a','od11a','od21a'

decoding_maps =[];
%specify results name
%res_name = 'object_decoding';
fname = '^s05wmean.*.nii$'; %'s05wres_accuracy_minus_chance.nii' ;

% load searchlight results

for sub_no = 1:length(fmri_subs)
    
    sub_id = fmri_subs{sub_no};
        
    if ~any(ismember(fmri_excluded_subs,sub_id))
        
        % load fMRI RDMs
        fmri_fnames = cellstr(spm_select('fplistrec',fullfile(fmri_path,sub_id, 'results','decoding','obj_crossdecoding'),fname));

        if any(cellfun(@exist,fmri_fnames))
            fprintf('Loading fMRI %s\n',sub_id);
            
           decoding_maps = cat(4, decoding_maps, spm_read_vols(spm_vol(fmri_fnames{1})));

            
        else
            fprintf('Results not complete for sub %s\n',sub_id);
            
        end
    end
end

%% write mean results without stats masking
hdr = spm_vol(fmri_fnames{1}); %  this should be an image with the same dimensionality as the searchlight results
%hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
hdr.descrip = sprintf('Searchlight decoding');
%hdr = rmfield(hdr, 'n');
hdr.fname = fullfile(out_dir,['pairwise_crossdecoding.nii']); %accuracy_minus_chance
mean_vol= mean(decoding_maps,4);
spm_write_vol(hdr, mean_vol);

%% find voxels where all subjects have values and compute stats only for that area 

sz = size(decoding_maps);
stats_mask = zeros(sz(1),sz(2),sz(3));

for vox = 1:(sz(1)*sz(2)*sz(3))
    
    
   [idx1,idx2,idx3] = ind2sub([sz(1),sz(2),sz(3)],vox);
   
   stats_mask(vox) = ~any(decoding_maps(idx1,idx2,idx3,:) ==0);
   
end

hdr = spm_vol(fmri_fnames{1}); %  this should be an image with the same dimensionality as the searchlight results
hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
hdr.descrip = sprintf('Mask for running statistics');
hdr = rmfield(hdr, 'n');
hdr.fname = fullfile(out_dir,['stats_mask_crossdecoding.nii']);
spm_write_vol(hdr, stats_mask);

% now mask the group results with the stats mask for running the stats 

decoding_maps_masked = [];

for sub= 1:size(decoding_maps,4)
    
      this_vol = decoding_maps(:,:,:,sub);
      this_vol(stats_mask==0) = NaN;
      decoding_maps_masked = cat(4,decoding_maps_masked,this_vol);
end 
      

%% compute stats 

nperm = 10000;
cluster_th = 0.01;
significance_th = 0.05;
tail = 'right';

[sig_searchlight_max,sig_searchlight_weigh,~,~,~,statmap] = permutation_cluster_1sample_weight_alld_less_mem (permute(decoding_maps_masked,[4 1 2 3]), nperm, cluster_th, significance_th, tail);

%% correct for multiple comparisons with FDR correction

[~, ~, ~, adj_p] = fdr_bh(statmap,0.05,'pdep');

maskvol = zeros(size(statmap)); 
maskvol(adj_p<0.05) = 1; 
hdr.fname = fullfile(out_dir,['pairwise_crossdecoding_fdr_corrected_p_0.05.nii']);
spm_write_vol(hdr,maskvol);
%% mask results volume with stats and write mean images 


hdr = spm_vol(fmri_fnames{1}); %  this should be an image with the same dimensionality as the searchlight results
hdr = rmfield(hdr, 'dt'); % get rid of scaling factors from the original image
hdr.descrip = sprintf('Searchlight decoding masked');
hdr = rmfield(hdr, 'n');
hdr.fname = fullfile(out_dir,['pairwise_crossdecoding_masked_p_0.01.nii']);
spm_write_vol(hdr, mean_vol.*sig_searchlight_max);


hdr.fname = fullfile(out_dir,['pairwise_crossdecoding_masked_weigh_p_0.01.nii']);
spm_write_vol(hdr, mean_vol.*sig_searchlight_weigh);

