function [boots] = bootstrap_fixed_1D_diff(data1, data2, time, nboot, statsInfo)
% calculates bootstrapping clusters finding the 95% confidence interval of onset and peak time 
% takes input: data: observations x variable1 (i.e. subject x time)
% time: time points info, e.g., -200:20:998
% nboot: number of bootstrapping
% statsInfo: stats settings for cluster based permutation test

if nargin < 5 %(default settings of cluster based permutation test)
    statsInfo.nperm = 10000;
    statsInfo.cluster_th = 0.001;
    statsInfo.significance_th = 0.001;
    statsInfo.tail = 'right';
end

%initialize
[nobs, nvar1, nvar2] = size(data1);

% find stimulus onset 
t_zero = find(time==0);
t_diff = diff(time);
t_diff = t_diff(1); 

%compute onset and peak for original data
%find peak
data1_1D = squeeze(mean(data1,1));
[~,I1] = max(data1_1D(:));
data2_1D = squeeze(mean(data2,1));
[~,I2] = max(data2_1D(:));
boots.peak_diff.orig = (I1-I2)*t_diff;


%find significance and onset (run cluster analysis)
[sigMax1] = permutation_cluster_weight_alld_no_disp(data1, statsInfo);
[sigMax2] = permutation_cluster_weight_alld_no_disp(data2, statsInfo);

I1 = ind2sub(size(data1_1D),min(find(sigMax1)));
while sigMax1(I1+1) == 0
    I1 = ind2sub(size(data_1D),I1+min(find(sigMax1(I1+1:end))));
end
I2 = ind2sub(size(data2_1D),min(find(sigMax2)));
while sigMax2(I2+1) == 0
    I2 = ind2sub(size(data_1D),I2+min(find(sigMax2(I2+1:end))));
end
boots.onset_diff.orig  = (I1-I2)*t_diff;
boots.mean1 = mean(data1,1);
boots.th1 = sigMax1;
boots.tndx1= find(sigMax1>0);
boots.mean2 = mean(data2,1);
boots.th2 = sigMax2;
boots.tndx2= find(sigMax2>0);

%if no significant points in original data
if isempty(sigMax1) | isempty(sigMax2)
    disp('No significant points found in one or both datasets');
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
    databoot1 = data1(bootsamples(i,:),:);
    databoot1_1D = squeeze(nanmean(databoot1,1));
    
    databoot2 = data2(bootsamples(i,:),:);
    databoot2_1D = squeeze(nanmean(databoot1,1));
    
    %find peak
    [~,I1] = max(databoot1_1D(:));
    [~,I2] = max(databoot2_1D(:));
    peak_tmp(i,:) = (I1-I2)*t_diff;
     
    %onset (run cluster analysis)
    [sigMax1] = permutation_cluster_weight_alld_no_disp(databoot1, statsInfo);
    [sigMax2] = permutation_cluster_weight_alld_no_disp(databoot2, statsInfo);
    %if significant points have been found
    I1 = ind2sub(size(databoot1_1D),min(find(sigMax1)));
    while sigMax1(I1+1) == 0
        I1 = ind2sub(size(databoot1_1D),I1+min(find(sigMax1(I1+1:end))));
    end
    I2 = ind2sub(size(databoot2_1D),min(find(sigMax2)));
    while sigMax2(I2+1) == 0
        I2 = ind2sub(size(databoot2_1D),I2+min(find(sigMax2(I2+1:end))));
    end
    onset_tmp2 = (I1-I2)*t_diff
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
    boots.peak_diff.boot = peak;
    boots.peak_diff.confidence95 = peakRow95;
    
else
    disp('Not enough points for peak 95% confidence interval');
end

if length(onset)>=100
    n1 = ceil(length(onset)*0.025);
    n2 = floor(length(onset)*0.975);
    onsetRow95(1) = onset(n1);
    onsetRow95(2) = onset(n2);
    boots.onset_diff.boot = onset;
    boots.onset_diff.confidence95 = onsetRow95;
    
else
    disp('Not enough points for onset 95% confidence interval');
end

% caluclate p-value for the difference by calculating how many values in
% the bootstrap distribution are equal to or exceed zero 
peak = [peak ;0];
onset = [onset; 0];

boots.peak_p = mean(peak<=0); 
boots.onset_p = mean(onset<=0); 
end