function [sigMax, sigWei, pValMax, pValWei, clusters] = permutation_cluster_weight_alld_no_disp(data, statsInfo)

if nargin < 2
 statsInfo.nperm = 1000;
 statsInfo.cluster_th = 0.05;
 statsInfo.significance_th = 0.05;
 statsInfo.tail = 'right';
end

%decide one-sided (right) or two-sided (both) test
if strcmp(statsInfo.tail,'right')
    func = '';
else
    func = 'abs';
end

%initialize
N = ndims(data); %data is observations x variable1 x variable2 x ... (supports all dimensions)
nobservations = size(data,1);
for n = 2:N
    nvariable(n-1) = size(data,n);
end
cln = repmat({':'},1,N-1); %to select the N-1 dimensions

%create permutation samples and convert them to pvalues (perms x variable1 x variable2 x ...)
StatMapPerm = single(zeros([statsInfo.nperm nvariable]));

%first permutation sample is original data
StatMapPerm(1,cln{:}) = mean(data,1) ./ std(data);

%perform permutations
parfor i = 2:statsInfo.nperm
    perm = single(sign(rand(nobservations,1)-0.5));
    %multiply by filling extra dimesions
    data_perm = repmat(perm,[1, nvariable]) .* data;
    permsample = mean(data_perm,1) ./ std(data_perm);
    StatMapPerm(i,:) = permsample(:); %does this for multiple dimensions: StatMapPerm(i,:,:) = mean(data_perm,1) ./ std(data_perm);
end

%convert to pvalues
eval([ 'StatMapPermPV = (statsInfo.nperm+1 - tiedrank(' func '(StatMapPerm)))/statsInfo.nperm;' ]);
StatMapPermPV = StatMapPermPV;

%find maximum cluster size and maximum weighted cluster for all permutation samples
[clusterMaxSize(1), clusterMaxWei(1), clusters, clustersize, clusterweight] = find_clusters_weight_alld(squeeze(StatMapPerm(1,cln{:})), squeeze(StatMapPermPV(1,cln{:})<=statsInfo.cluster_th));
parfor i = 2:statsInfo.nperm
    [clusterMaxSize(i), clusterMaxWei(i)] = find_clusters_weight_alld(squeeze(StatMapPerm(i,cln{:})), squeeze(StatMapPermPV(i,cln{:})<=statsInfo.cluster_th));
end

%find cluster threshold
clusterMaxSize_sorted = sort(clusterMaxSize, 'descend');
clusterMaxWei_sorted = sort(clusterMaxWei, 'descend');
th_max = clusterMaxSize_sorted( statsInfo.nperm*statsInfo.significance_th );
th_wei = clusterMaxWei_sorted( statsInfo.nperm*statsInfo.significance_th );

%find significant variables
if length(nvariable) == 1
    sigMax = zeros(nvariable,1);
    sigWei = zeros(nvariable,1);
else
    sigMax = zeros(nvariable);
    sigWei = zeros(nvariable);
end

%apply threshold on found clusters
sigMax([clusters{clustersize>th_max}]) = 1;
sigWei([clusters{clusterweight>th_wei}]) = 1;
if ~isempty (clustersize)
    pValMax = ( find(clusterMaxSize_sorted == clusterMaxSize(1), 1, 'first') ) / length(clusterMaxSize_sorted);
    pValWei = ( find(clusterMaxWei_sorted == clusterMaxWei(1), 1, 'first') ) / length(clusterMaxWei_sorted);
else
    pValMax = NaN;
    pValWei = NaN;
end

end