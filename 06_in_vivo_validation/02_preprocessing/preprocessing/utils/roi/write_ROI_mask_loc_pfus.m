% function  write_ROI_mask_loc_pfus(cfg, i_sub, p, df)
%
% This code can be used to compute one combined ROI masks for the regions LOC and pfus which are defined 
% by taking the cluster peaks from the localizer tak of the experiment.
% We do a number of things:
% 1. We convert the cluster peaks from individual space to voxel space
% 2. We get the voxels that lie inside the statistical cluster defined in
%    the localizer contrast
% 3. We draw a sphere around the two cluster peaks and take all voxels
%    inside that sphere into our ROI mask 
% 4. We combine the two clusters into a combined LOC + pFus mask and write
%    the mask 
%
% INPUT:
%   cfg: a config structure with subject specific and experiment specific
%   details obtained from config_subjects_objdraw.m
%   i_sub: the number of the subject for which the masks should be written
%   p: p-value with which the cluster from the loaclizer contrast should be
%   thresholded  (default=0.0001)
%   df: the corresponding degrees of freedom for the cluster statistic
%   (default=294)
%   radius: radius for the spheres that constrain the ROIs (default = 10)
%
function write_ROI_mask_loc_pfus(cfg,i_sub,p,df, radius)

% load .nii image for hdr information
hdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'results','GLM','localizer','beta_0001.nii'));

% get coordinates for the rois 
loc_R = [cfg.sub(i_sub).roi.loc_right.position 1]';
pfus_R = [cfg.sub(i_sub).roi.pfus_right.position 1]';
loc_L = [cfg.sub(i_sub).roi.loc_left.position 1]';
pfus_L = [cfg.sub(i_sub).roi.pfus_left.position 1]';

% transform coordinates into voxelspace
loc_R = round(hdr.mat\loc_R);
loc_R(4) = [];
pfus_R = round(hdr.mat\pfus_R);
pfus_R(4) = [];
loc_L = round(hdr.mat\loc_L);
loc_L(4) = [];
pfus_L = round(hdr.mat\pfus_L);
pfus_L(4) = [];

% load the contrast t-values
hdr = spm_vol(fullfile(cfg.sub(i_sub).dir,'results','GLM','localizer','spmT_0001.nii'));
vol = spm_read_vols(hdr); 

%check if p and df are given if not assign default value
if ~exist('p','var'), p=0.0001, end 
if ~exist('df','var'), df=294, end 

%threshold the t-values with a given p and df
T_thresh = tinv(1-p,df); % p , df repectively 

masked_vol = vol>T_thresh; 

% get all cluster ids
[cluster_vol,num] = spm_bwlabel(double(masked_vol),18); %18 is connectivity criterium 

%find the cluster id for the peaks
loc_R_cluster_id = cluster_vol(loc_R(1),loc_R(2),loc_R(3));
pfus_R_cluster_id = cluster_vol(pfus_R(1),pfus_R(2),pfus_R(3));
loc_L_cluster_id = cluster_vol(loc_L(1),loc_L(2),loc_L(3));
pfus_L_cluster_id = cluster_vol(pfus_L(1),pfus_L(2),pfus_L(3));

%warning when any of the cluster id = 0 
if any([loc_R_cluster_id pfus_R_cluster_id loc_L_cluster_id pfus_L_cluster_id] == 0)
    warning('One or more of the peaks do not belong to a cluster - check the peak coordinates!')
end 

% assign radius for the spheres if not given - default = 5
if ~exist('radius','var'), radius = 6, end 

%define the sphere independent of the exact position
sz = size(vol);
center_coor = round(sz/2);
center_ind = sub2ind(sz,center_coor(1),center_coor(2),center_coor(3));
[x,y,z] = ndgrid(1:sz(1),1:sz(2),1:sz(3));
sphere_mask = ((x-center_coor(1)).^2 + (y-center_coor(2)).^2 + (z-center_coor(3)).^2) < radius.^2;
mask_ind = find(sphere_mask) - center_ind;

%get centercoordinates for the ROI spheres
loc_R_centerind = sub2ind(sz,loc_R(1),loc_R(2),loc_R(3)); %center index
pfus_R_centerind = sub2ind(sz,pfus_R(1),pfus_R(2),pfus_R(3));
loc_L_centerind = sub2ind(sz,loc_L(1),loc_L(2),loc_L(3)); %center index
pfus_L_centerind = sub2ind(sz,pfus_L(1),pfus_L(2),pfus_L(3));

% get the voxelindices for the voxels inside the sphere and that belong to
% a statistical cluster
locind_R = intersect(find(cluster_vol==loc_R_cluster_id),loc_R_centerind+mask_ind);
pfusind_R = intersect(find(cluster_vol==pfus_R_cluster_id),pfus_R_centerind+mask_ind);
union_R = (union(locind_R, pfusind_R));
locind_L = intersect(find(cluster_vol==loc_L_cluster_id),loc_L_centerind+mask_ind);
pfusind_L = intersect(find(cluster_vol==pfus_L_cluster_id),pfus_L_centerind+mask_ind);
union_L = (union(locind_L, pfusind_L));



if ~isdir(fullfile(cfg.sub(i_sub).dir,'roi')), mkdir(fullfile(cfg.sub(i_sub).dir,'roi')), end 

maskhdr = spm_vol(fullfile(cfg.sub(i_sub).dir, 'results','GLM','first_level_denoise','mask.nii'));
maskvol = zeros(sz);
maskvol(union_R) = 1;
maskvol(union_L) = 1;
maskhdr.fname = fullfile(cfg.sub(i_sub).dir,'roi','combined_loc_fus_mask.nii');
spm_write_vol(maskhdr,maskvol);

end 