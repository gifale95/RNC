%% evaluate decision values from ROIs 

clear all


set(0, 'defaultaxesfontsize', 14, 'defaultaxesfontweight', 'bold', ...
    'defaultlinelinewidth', 3)  

sub_id = 'sub02'; 

outdir = ['/Users/johannessinger/scratch/dfg_projekt/WP1/derived/pilot/',sub_id, '/results/decoding/crossdecoding_controversial_hrf_fitting/roi'];
roi_dir = ['/Users/johannessinger/scratch/dfg_projekt/WP1/derived/pilot',sub_id, '/roi']; 
load(fullfile(outdir,'res_decision_values.mat'));

manmade_manmade_vals = zeros(1,length(results.decision_values.output)); 
manmade_natural_vals = zeros(1,length(results.decision_values.output)); 
natural_manmade_vals = zeros(1,length(results.decision_values.output)); 
natural_natural_vals = zeros(1,length(results.decision_values.output)); 


for i = 1:length(results.decision_values.output)
    
    these_dec_vals = results.decision_values.output{i};
    
    manmade_manmade_vals(i) = mean(these_dec_vals(1:15)); 
    manmade_natural_vals(i) = mean(these_dec_vals(16:30)); 
    natural_manmade_vals(i) = mean(these_dec_vals(31:45));
    natural_natural_vals(i) = mean(these_dec_vals(46:end));
end 

%% plot 


figure 

all_mean_vals = cat(2, manmade_manmade_vals', manmade_natural_vals',natural_manmade_vals',natural_natural_vals')'


bar(all_mean_vals);
ylim([-1 1]); 
    legend({'EVC', 'LOC','PPA'} ,'Location','southwest')
xticklabels({'Low-manmade High-manmade'; 'Low-manmade High-natural'; 'Low-natural High-manmade';'Low-natural High-natural'})