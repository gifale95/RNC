%% get slicetiming, sliceorder, microtime onset and microtime resolution

clear all 
clc

addpath('/Users/johannessinger/Documents/cloud_Berlin/Projekte/dfg/WP1/analysis/spm12')

input_dir = '/Users/johannessinger/Documents/cloud_Berlin/Projekte/dfg/WP1/data/CCNB_9403';
P = spm_select('FPListRec',input_dir,'[0-9]'); 
dicom = P(1,:);
[slice_order, TR, nslices, slice_order_type, slice_timing] = autodetect_sliceorder(dicom, 1);
[microtime_onset, microtime_resolution, refslice] = gen_microtime_onset(slice_order, 'middle');

save('slicetiming.mat', 'slice_timing');
