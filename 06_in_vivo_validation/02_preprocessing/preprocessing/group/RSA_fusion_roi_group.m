%% distance to hyperplane analysis 
clear all

for i_sub = 1:9
    
res_path = fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived','exp',['sub0' num2str(i_sub)],'results','RSA_pearson','roi');

load(fullfile(res_path,'res_other_average.mat'));

for roi = 1:3
    
    RDMs_fmri(:,i_sub,roi) = squareform(1-results.other_average.output{roi}); 

end 
end 

load(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived', 'eeg','rdm_pearson_eeg_all_subs_categorization.mat'))

%% get the mean EEG RDV
eeg_RDVs = []; 

for sub = 1:30
    for t = 1:200
        eeg_RDV = squeeze(rdm_eeg_all_subs(sub,:,:,t)).';
        m  = tril(true(size(eeg_RDV)),-1);
        eeg_RDVs(:,sub,t)  = eeg_RDV(m).';
    end
end

%% compute dth fusion 

for roi = 1:3
    
    RDV_fmri = mean(RDMs_fmri(:,:,roi),2);
        for t = 1:size(rdm_eeg_all_subs,4)
            
            RDV_eeg = nanmean(eeg_RDVs(:,:,t),2);
            dth_fusion(t,roi) = corr(RDV_fmri,RDV_eeg);
        end
end

%% compute dth fusion with single subs in EEG

single_sub_fusion = [];

for roi = 1:3
    
    RDV_fmri = mean(RDMs_fmri(:,:,roi),2);
    for sub = 1:30
        for t = 1:size(rdm_eeg_all_subs,4)
            
            RDV_eeg = eeg_RDVs(:,sub,t);
            single_sub_fusion(sub,t,roi) = corr(RDV_fmri,RDV_eeg);
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
ylim([-0.1 1])
legend('EVC','PPA', 'LOC')

%% plot fusion with single subjects 

figure 
plot(nanmean(single_sub_fusion(:,:,1)))
hold on 
plot(nanmean(single_sub_fusion(:,:,2)))
plot(nanmean(single_sub_fusion(:,:,3)))
xlim([20 110])
xticks([20:20:110])
xticklabels([-100:100:400])
legend('EVC','PPA', 'LOC')

print(fullfile('/Users/johannessinger/scratch/dfg_projekt/WP1/derived','exp','group', ['rsa_fusion_single_sub_eeg_zoom.jpg']), ...
        '-djpeg', '-r300')
%%
figure 
for sub = 1:30 
plot(nanmean(single_sub_fusion(:,:,1)))
hold on 
plot(nanmean(single_sub_fusion(:,:,2)))
plot(nanmean(single_sub_fusion(:,:,3)))
xlim([10 200])
xticks([20:20:200])
xticklabels([-100:100:800])
end 
legend('EVC','PPA', 'LOC')
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


