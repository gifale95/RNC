function [boots] = bootstrap_fixed_1D(data, time, nboot, statsInfo)
% calculates bootstrapping clusters finding the 95% confidence interval of onset and peak time 
% takes input: data: observations x variable1 (i.e. subject x time)
% time: time points info, e.g., -200:20:998
% nboot: number of bootstrapping
% statsInfo: stats settings for cluster based permutation test

if nargin < 4 %(default settings of cluster based permutation test)
    statsInfo.nperm = 10000;
    statsInfo.cluster_th = 0.001;
    statsInfo.significance_th = 0.001;
    statsInfo.tail = 'right';
end

%initialize
[nobs, nvar1, nvar2] = size(data);

%compute onset and peak for original data
%find peak
data_1D = squeeze(mean(data,1));
[~,I] = max(data_1D(:));
boots.peak.orig = time(I);

%find significance and onset (run cluster analysis)
[sigMax] = permutation_cluster_weight_alld_no_disp(data, statsInfo);

I1 = ind2sub(size(data_1D),min(find(sigMax)));
while sigMax(I1+1) == 0 
    I1 = ind2sub(size(data_1D),I1+min(find(sigMax(I1+1:end))));
end 
boots.onset.orig  = time(I1);
boots.mean = mean(data,1);
boots.th = sigMax;
boots.tndx = find(sigMax>0);

%if no significant points in original data
if isempty(sigMax)
    disp('No significant points found');
    return
end

%make bootstrap samples
bootsamples = randi(nobs,nboot,nobs);

%compute onset and peak for bootstrap samples
parfor i = 1:nboot
    % nboot
    
    if ~rem(i,5)
        disp(['Process bootstrap samples: ' num2str(i) ' out of ' num2str(nboot)]);
    end
    
    %create bootstrap sample
    databoot = data(bootsamples(i,:),:);
    databoot_1D = squeeze(nanmean(databoot,1));
    
    %find peak
    [~,I] = max(databoot_1D(:));
    peak_tmp(i,:) = [time(I)];
     
    %onset (run cluster analysis)
    [sigMax] = permutation_cluster_weight_alld_no_disp(databoot, statsInfo);
    %if significant points have been found
    I1 = ind2sub(size(databoot_1D),min(find(sigMax)));
    while sigMax(I1+1) == 0 
    I1 = ind2sub(size(data_1D),I1+min(find(sigMax(I1+1:end))));
    end 
    onset_tmp2 = time(I1);
    if ~isempty(onset_tmp2)
        onset_tmp(i,:) = onset_tmp2;
    else
        onset_tmp(i,:) = [nan];
    end
    
end

%remove nan points
onset_tmp(isnan(onset_tmp(:,1)),:)=[];

%sort
peak = sort(peak_tmp(:,1));
onset = sort(onset_tmp(:,1));

%estimate 95% confidence intervals
if length(peak)>=100
    n1 = ceil(length(peak)*0.025);
    n2 = floor(length(peak)*0.975);
    peakRow95(1) = peak(n1);
    peakRow95(2) = peak(n2);
    boots.peak.boot = peak;
    boots.peak.confidence95 = peakRow95;
    
else
    disp('Not enough points for peak 95% confidence interval');
end

if length(onset)>=100
    n1 = ceil(length(onset)*0.025);
    n2 = floor(length(onset)*0.975);
    onsetRow95(1) = onset(n1);
    onsetRow95(2) = onset(n2);
    boots.onset.boot = onset;
    boots.onset.confidence95 = onsetRow95;
    
else
    disp('Not enough points for onset 95% confidence interval');
end

end