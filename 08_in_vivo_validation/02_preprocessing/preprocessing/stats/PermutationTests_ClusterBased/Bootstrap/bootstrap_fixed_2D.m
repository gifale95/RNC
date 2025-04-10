function [boots] = bootstrap_fixed_2D(data, time, nboot, statsInfo)
% calculates bootstrapping clusters finding the 95% confidence interval of onset and peak time 
% takes input: data: observations x variable1 x variable2 (i.e. subject x time x time)
% time: time points info, e.g., -200:20:998
% nboot: number of bootstrapping
% statsInfo: stats settings for cluster based permutation test

if nargin < 4 %(default settings of cluster based permutation test)
    statsInfo.nperm = 10000;
    statsInfo.cluster_th = 0.05;
    statsInfo.significance_th = 0.05;
    statsInfo.tail = 'right';
end

%initialize
[nobs, nvar1, nvar2] = size(data);

%compute onset and peak for original data
%find peak
data_2D = squeeze(mean(data,1));
[~,I] = max(data_2D(:));
[I1, I2]= ind2sub(size(data_2D),I);
boots.peakRow.orig = time(I1);
boots.peakColumn.orig = time(I2);

%find significance and onset (run cluster analysis)
[sigMax] = permutation_cluster_weight_alld_no_disp(data, statsInfo);

[i1, i2] = ind2sub(size(data_2D),min(find(sigMax)));
boots.onsetRow.orig = time(i1);
boots.onsetColumn.orig  = time(i2);
boots.onsetDiff.orig = boots.onsetRow.orig - boots.onsetColumn.orig;
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
    databoot = data(bootsamples(i,:),:,:);
    databoot_2D = squeeze(nanmean(databoot,1));
    
    %find peak
    [~,I] = max(databoot_2D(:));
    [I1, I2]= ind2sub(size(databoot_2D),I);
    peak(i,:) = [time(I1), time(I2)];
     
    %onset (run cluster analysis)
    [sigMax] = permutation_cluster_weight_alld_no_disp(databoot, statsInfo);
    %if significant points have been found
    [i1, i2] = ind2sub(size(databoot_2D),min(find(sigMax)));
    onset_tmp = [time(i1),time(i2)];
    if ~isempty(onset_tmp)
        onset(i,:) = onset_tmp;
    else
        onset(i,:) = [nan,nan];
    end
    
end

%remove nan points
onset(isnan(onset(:,1)),:)=[];


%sort
peakDiff = sort(peak(:,1) - peak(:,2));
onsetDiff = sort(onset(:,1) - onset(:,2));

peakRow = sort(peak(:,1));
onsetRow = sort(onset(:,1));
peakColumn = sort(peak(:,2));
onsetColumn = sort(onset(:,2));

%estimate 95% confidence intervals
if length(peak)>=100
    n1 = ceil(length(peakRow)*0.025);
    n2 = floor(length(peakRow)*0.975);
    peakRow95(1) = peakRow(n1);
    peakRow95(2) = peakRow(n2);
    boots.peakRow.boot = peakRow;
    boots.peakRow.confidence95 = peakRow95;
    
    n1 = ceil(length(peakColumn)*0.025);
    n2 = floor(length(peakColumn)*0.975);
    peakColumn95(1) = peakColumn(n1);
    peakColumn95(2) = peakColumn(n2);
    boots.peakColumn.boot = peakColumn;
    boots.peakColumn.confidence95 = peakColumn95;
    
    
    n1 = ceil(length(peakDiff)*0.025);
    n2 = floor(length(peakDiff)*0.975);
    peakDiff95(1) = peakDiff(n1);
    peakDiff95(2) = peakDiff(n2);
    boots.peakDiff.boot = peakDiff;
    boots.peakDiff.confidence95 = peakDiff95;
    
else
    disp('Not enough points for peak 95% confidence interval');
end

if length(onset)>=100
    n1 = ceil(length(onsetRow)*0.025);
    n2 = floor(length(onsetRow)*0.975);
    onsetRow95(1) = onsetRow(n1);
    onsetRow95(2) = onsetRow(n2);
    boots.onsetRow.boot = onsetRow;
    boots.onsetRow.confidence95 = onsetRow95;
    
    n1 = ceil(length(onsetColumn)*0.025);
    n2 = floor(length(onsetColumn)*0.975);
    onsetColumn95(1) = onsetColumn(n1);
    onsetColumn95(2) = onsetColumn(n2);
    boots.onsetColumn.boot = onsetColumn;
    boots.onsetColumn.confidence95 = onsetColumn95;
    
    n1 = ceil(length(onsetDiff)*0.025);
    n2 = floor(length(onsetDiff)*0.975);
    onsetDiff95(1) = onsetDiff(n1);
    onsetDiff95(2) = onsetDiff(n2);
    boots.onsetDiff.boot = onsetDiff;
    boots.onsetDiff.confidence95 = onsetDiff95;
else
    disp('Not enough points for onset 95% confidence interval');
end

end