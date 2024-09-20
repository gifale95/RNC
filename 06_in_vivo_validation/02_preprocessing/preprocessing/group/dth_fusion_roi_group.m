%% distance to hyperplane analysis 
clear all

for i_sub = 1:9
    
res_path = fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived','exp',['sub0' num2str(i_sub)],'results','decoding','manmade_natural_hrf_fitting','roi');

load(fullfile(res_path,'res_mean_decision_values.mat'));

for roi = 1:3
    
    distances_fmri(:,i_sub,roi) = results.mean_decision_values.output{roi}; 

end 
end 

load(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived', 'behav','RT_all_subjects_5_35_categorization.mat'), 'RTs')
load(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived', 'eeg','distances_all_subjects.mat'))


%% normalize the RTs and average them 
% 
%norm_RTs = zscore(RTs, 1,2); 

mean_RTs = nanmedian(RTs,1); 

%% compute dth fusion 

for roi = 1:3
    
    mean_distances_fmri = mean(distances_fmri(:,:,roi),2); 
    for t = 1:size(distances_all_subjects,3)
        
        mean_distances_eeg = nanmean(distances_all_subjects(:,:,t)); 
        dth_fusion(t,roi) = corr(mean_distances_eeg',mean_distances_fmri); 
    end 
    end 

%% compute dth fusion with single subs in EEG

for roi = 1:3 
    
    mean_distances_fmri = mean(cat(2,distances_fmri(:,:,roi)),2); 
    for sub = 1:30
    for t = 1:size(distances_all_subjects,3)
        
        mean_distances_eeg = distances_all_subjects(sub,:,t); 
        dth_fusion(t,sub,roi) = corr(mean_distances_eeg',mean_distances_fmri); 
    end 
    end 
end 

%% compute only behav eeg correlation 


for sub = 1:30
    for t = 1:size(distances_all_subjects,3)
        
        mean_distances_eeg = distances_all_subjects(sub,:,t);
        dth_eeg(t,sub) = corr(mean_distances_eeg',mean_RTs');
    end
end

figure
plot(nanmean(dth_eeg.^2,2))
hold on
plot(nanmean(dth_eeg,2))
xticks([0:20:200])
xticklabels([-200:100:800])
ylim([-0.2 0.1])
yline(0)

%% compute the shared variance between fMRI, EEG and behavior 

for roi = 1:3
    
    mean_distances_fmri = mean(distances_fmri(:,:,roi),2); 

    for t = 1:size(distances_all_subjects,3)
        
        mean_distances_eeg = nanmean(distances_all_subjects(:,:,t)); 
        [~,~,~,~,r2_behav_eeg] = regress(mean_distances_eeg', [ones(size(mean_distances_fmri,1),1) mean_RTs']); 
        [~,~,~,~,r2_fmri_eeg] = regress(mean_distances_eeg', [ones(size(mean_distances_fmri,1),1) mean_distances_fmri]); 
        [~,~,~,~,r2_full] = regress(mean_distances_eeg', [ones(size(mean_distances_fmri,1),1), mean_distances_fmri, mean_RTs']);
        
        dth_shared(t,roi) = r2_fmri_eeg(1)+r2_behav_eeg(1)-r2_full(1); 
        dth_behav_eeg(t,roi) = r2_behav_eeg(1); 
    end 
end

%% compute the shared variance between fMRI, EEG and behavior with single subjects 

for roi = 1:3
    
    mean_distances_fmri = mean(distances_fmri(:,:,roi),2); 
    
    for sub = 1:30
    for t = 1:size(distances_all_subjects,3)
        
        mean_distances_eeg = distances_all_subjects(sub,:,t); 
        [~,~,~,~,r2_behav_eeg] = regress(mean_distances_eeg', [ones(size(mean_distances_fmri,1),1) mean_RTs']); 
        [~,~,~,~,r2_fmri_eeg] = regress(mean_distances_eeg', [ones(size(mean_distances_fmri,1),1) mean_distances_fmri]); 
        [~,~,~,~,r2_full] = regress(mean_distances_eeg', [ones(size(mean_distances_fmri,1),1), mean_distances_fmri, mean_RTs']);
        
        dth_shared(t,sub,roi) = r2_fmri_eeg(1)+r2_behav_eeg(1)-r2_full(1); 
        dth_behav_eeg(t,sub,roi) = r2_behav_eeg(1); 
    end 
    end
end

%% plot fusion with mean on both sides 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3)

figure 

plot(dth_fusion(:,1),'Color',[0 0.4470 0.7410])
hold on 
plot(dth_fusion(:,2),'Color',[0.8500 0.3250 0.0980])
plot(dth_fusion(:,3),'Color', [0.9290 0.6940 0.1250])
xticks([0:20:200])
xticklabels([-200:100:800])
ylim([-0.1 0.5])
legend('EVC','PPA', 'LOC')

%% plot fusion with single subjects 

figure 
plot(nanmean(dth_fusion(:,:,1),2))
hold on 
plot(nanmean(dth_fusion(:,:,2),2))
plot(nanmean(dth_fusion(:,:,3),2))
xticks([0:20:200])
xticklabels([-200:100:800])
legend('EVC','PPA', 'LOC')

print(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived','exp','group', ['dth_fusion_single_sub_eeg.jpg']), ...
        '-djpeg', '-r300')

%% only variance partitioning 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3)

figure 
%h = area(dth_only_behav(:,1));
%h.FaceColor = [0.9 0.9 0.9];
hold on
plot(dth_behav_eeg(:,1),'Color',[0 0.4470 0.7410])
plot(dth_behav_eeg(:,2),'Color',[0.8500 0.3250 0.0980])
plot(dth_behav_eeg(:,3),'Color', [0.9290 0.6940 0.1250])
plot(dth_shared(:,1),'Color',[0 0.4470 0.7410], 'Linestyle',':')
plot(dth_shared(:,2),'Color',[0.8500 0.3250 0.0980],'Linestyle',':')
plot(dth_shared(:,3),'Color', [0.9290 0.6940 0.1250],'Linestyle',':')
xticks([0:20:200])
xticklabels([-200:100:800])
ylim([-0.05 0.2])
legend('EVC','PPA', 'LOC')

%% only variance partitioning with single subjects 

set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3)

figure 
h = area(nanmean(dth_behav_eeg(:,:,1),2));
h.FaceColor = [0.9 0.9 0.9];
hold on
plot(nanmean(dth_shared(:,:,1),2),'Color',[0 0.4470 0.7410])
plot(nanmean(dth_shared(:,:,2),2),'Color',[0.8500 0.3250 0.0980])
plot(nanmean(dth_shared(:,3),2),'Color', [0.9290 0.6940 0.1250])
xlim([10 120])
xticks([20:20:120])
xticklabels([-100:100:400])
ylim([-0.01 0.08])
legend('Behavior-EEG', 'EVC','PPA', 'LOC')
print(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived','exp','group', ['dth_variance_partitioning_ROI.jpg']), ...
        '-djpeg', '-r300')

%% full variance  

figure 
%h = area(dth_only_behav(:,1));
%h.FaceColor = [0.9 0.9 0.9];
hold on
plot(dth_full(:,1),'Color',[0 0.4470 0.7410])
plot(dth_full(:,2),'Color',[0.8500 0.3250 0.0980])
plot(dth_full(:,3),'Color', [0.9290 0.6940 0.1250])
xticks([0:20:200])
xticklabels([-200:100:800])
ylim([-0.01 0.5])
legend('EVC','PPA', 'LOC')

%% compute variance explained by behavior 

for roi = 1:3 
    
    mean_distances_fmri = mean(cat(2,distances_fmri(:,:,roi)),2); 
    for sub = 1:30
    for t = 1:size(distances_all_subjects,3)
        
        mean_distances_eeg = distances_all_subjects(sub,:,t); 
        this_r = correlate([mean_distances_eeg' mean_distances_fmri mean_RTs'],'type','Spearman','method','semipartialcorr');
        r_behav(t,sub,roi) = this_r(2,1);
    end 
    end 
end 

%% plot 

figure 
plot(nanmean(r_behav(:,:,1),2))
hold on 
plot(nanmean(r_behav(:,:,2),2))
plot(nanmean(r_behav(:,:,3),2))
xticks([0:20:200])
xticklabels([-200:100:800])
legend('EVC','PPA', 'LOC')


