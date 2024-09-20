function run_contrasts_rcor(results_dir,control_condition_names,challenge_condition_names);

data = load(fullfile(results_dir,'SPM.mat'));
SPM = data.SPM;
control_weights = [];
challenge_weights = [];
nuissance_weights = zeros(1,36);
for run = 1:length(SPM.Sess)
    
    this_conds = cellfun(@char,{SPM.Sess(run).U.name},'UniformOutput',false);
    this_weights = ismember(this_conds,control_condition_names);
    control_weights = [control_weights, this_weights,nuissance_weights];
    this_weights = ismember(this_conds,challenge_condition_names);
    challenge_weights = [challenge_weights, this_weights,nuissance_weights];
end
difference_weights =control_weights*-1+challenge_weights;
all_bigger_baseline_weights = control_weights+challenge_weights;


%Set contrasts
clear matlabbatch
matlabbatch{1}.spm.stats.con.spmmat = {fullfile(results_dir,'SPM.mat')};
matlabbatch{1}.spm.stats.con.consess{1}.tcon.name = 'control';
matlabbatch{1}.spm.stats.con.consess{1}.tcon.weights = control_weights./(length(SPM.Sess)/2);
matlabbatch{1}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.consess{2}.tcon.name = 'challenge';
matlabbatch{1}.spm.stats.con.consess{2}.tcon.weights = challenge_weights./(length(SPM.Sess)/2);
matlabbatch{1}.spm.stats.con.consess{2}.tcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.consess{3}.tcon.name = 'challenge vs. control';
matlabbatch{1}.spm.stats.con.consess{3}.tcon.weights = difference_weights./length(SPM.Sess);
matlabbatch{1}.spm.stats.con.consess{3}.tcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.consess{4}.tcon.name = 'all vs. baseline';
matlabbatch{1}.spm.stats.con.consess{4}.tcon.weights = all_bigger_baseline_weights/length(SPM.Sess);
matlabbatch{1}.spm.stats.con.consess{4}.tcon.sessrep = 'none';
matlabbatch{1}.spm.stats.con.delete = 1;

spm_jobman('run',matlabbatch)
end