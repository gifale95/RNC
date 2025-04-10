%% load the Wang Atlas 2015 definitions for V1,V2,V3 and V4 and create ROI masks from there

clear all

roi_labels = [1:25];
labels = readlines('/Users/johannessinger/Documents/cloud_Berlin/Projekte/dfg/WP1/analysis_tools/wang_atlas/ProbAtlas_v4/ROIfiles_Labeling.txt')
labels = labels(1:end-1);
ct = 1; 
for roi = 2:length(labels)
    
    split_string = split(labels(roi));
    roi_names(ct) = {char(split_string(end-1))};
    ct = ct+1;
end 

for roi = 1:length(roi_labels)
    
    roi_label = roi_labels(roi);
    
    all_lh_vol = [];
    all_rh_vol = [];
    
    for idx = 1:length(roi_label)
        
    lh_vol = spm_read_vols(spm_vol(fullfile('/Users/johannessinger/Documents/cloud_Berlin/Projekte/dfg/WP1/analysis_tools/wang_atlas/ProbAtlas_v4/subj_vol_all',['perc_VTPM_vol_roi',num2str(roi_label(idx)),'_lh.nii'])));
    rh_vol =  spm_read_vols(spm_vol(fullfile('/Users/johannessinger/Documents/cloud_Berlin/Projekte/dfg/WP1/analysis_tools/wang_atlas/ProbAtlas_v4/subj_vol_all',['perc_VTPM_vol_roi',num2str(roi_label(idx)),'_rh.nii']))); 
    
    all_lh_vol = cat(4,all_lh_vol,lh_vol>20); 
    all_rh_vol = cat(4,all_rh_vol,rh_vol>20); 

    end 

    
mask_lh_vol = sum(all_lh_vol,4)>0; 
mask_rh_vol = sum(all_rh_vol,4)>0;
full_mask_vol = mask_lh_vol+mask_rh_vol; 
hdr = spm_vol(fullfile('/Users/johannessinger/Documents/cloud_Berlin/Projekte/dfg/WP1/analysis_tools/wang_atlas/ProbAtlas_v4/subj_vol_all',['perc_VTPM_vol_roi',num2str(roi),'_lh.nii']));
hdr.fname = fullfile('/Users/johannessinger/scratch/rcor_collab/derived/roi/wang_roi_masks',[roi_names{roi},'_mask.nii']);
spm_write_vol(hdr,full_mask_vol);

    
end 