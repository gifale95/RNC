function [clusterSizeMax, clusterWeiMax, clusters, clusterSize, clusterWeight] = find_clusters_weight_alld(rf_data, rf)
% Finds clusters in random field of any dimension and then computes 
% (1) maximum size of the found clusters 
% (2) maximum weight of the found clusters
% 
% INPUTS:
%   rf_data: data of random field (array of any dimension)
%   rf: logical index of suprathrehold voxels in random field (array of any dimension)
%
% Author: Dimitrios Pantazis, 7/9/2014
% (modified): Siying Xie, August 2020

%initialize
dim = length(size(rf)); %random field dimension
nvoxels = numel(rf); %number of voxels
ndx = find(rf==true); %index of suprathreshold voxels
nsupvoxels = length(ndx); %number of suprathreshold voxels
i = 0; %no clusters at the begining
clear clusters clusterweight clustersize;
clusters{1} = []; %empty cluster
clusterWeight = [];
clusterSize = [];
supvoxelfound = zeros(nvoxels,1); %no voxels assigned to clusters yet

%if no suprathreshold voxels exist
if isempty(ndx)
    clusterSizeMax = 0;
    clusterWeiMax = 0;
    clusters = {[]};
    return
end

%template neighbour coordinates for suprathreshold voxels
expndx = rem(floor([0:3^dim-1]' * 3.^(-dim+1:0)),3) - 1; %build a list of neighbor indexes
s = sum(abs(expndx),2);
expndx(s == 0 | s == dim,:) = []; %remove points with no common edge (all dimensions change) or same point
nv = size(expndx,1); %number of neighbours

for k = 1:nsupvoxels %for all supratheshold voxels

    svox_ndx = ndx(k); %index of suprathreshold voxel
    
    %if voxel already belongs to a cluster 
    if supvoxelfound(svox_ndx) == 1
        continue
    end

    %create new cluster
    i = i+1;
    clusters{i} = svox_ndx; %new cluster
    supvoxelfound(svox_ndx) = 1; %found this voxel
    cluster_exp = svox_ndx; %expand cluster
    
    while ~isempty(cluster_exp) %while the cluster keeps expanding

        %find neighbours
        v = ind2sub_v(size(rf),cluster_exp); %index of new points
        vexp = reshape(repmat(v,1,nv)',size(v,2),size(v,1)*nv)'; %repeat points nv times rowwise
        vexp = vexp + repmat(expndx,size(v,1),1); %coordinates of new points
        vexp = unique(vexp,'rows'); %keep unique points
        
        %remove outside points
        [ndx1,~] = find(vexp==0); %at 0
        [ndx2,~] = find(vexp - ones(size(vexp,1),1)*(size(rf)+1)==0); %above max dim
        vexp([ndx1; ndx2],:) = [];
        
        %expand cluster
        cluster_exp = sub2ind_v(size(rf),vexp'); %convert to linear index
        cluster_exp(supvoxelfound(cluster_exp)==1)=[]; %remove already found voxels
        cluster_exp = cluster_exp(rf(cluster_exp)==true); %keep only suprathreshold voxels
        cluster_exp = setdiff(cluster_exp,clusters{i}); %keep only voxels that don't already exist in cluster
        clusters{i} = [clusters{i} cluster_exp]; %expand cluster
        supvoxelfound(cluster_exp) = 1; %found voxels 
    end
end

nclusters = size(clusters,2);
clusterSize = NaN(1,nclusters);
clusterWeight = NaN(1,nclusters);

%get the size of the first found cluster
clusterSizeMax = size(clusters{1},2);
clusterSize(1) =  size(clusters{1},2);
%get the weight of the first found cluster (i.e., take the values of cluster into consideration)
clusterWeiMax = sum(rf_data(clusters{1}));
clusterWeight(1) = sum(rf_data(clusters{1}));

for i = 2:nclusters
    %get max weights of the found clusters
    clusterWeiMax = max(clusterWeiMax, sum(rf_data(clusters{i})));
    clusterWeight(i) = sum(rf_data(clusters{i}));
    
    %get max size of the found clusters
    clusterSizeMax = max(clusterSizeMax, size(clusters{i},2));
    clusterSize(i) =  size(clusters{i},2);
end

end