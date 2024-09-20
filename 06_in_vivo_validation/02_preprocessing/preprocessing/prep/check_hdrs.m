function hdr = check_hdrs(hdr)
%%%%%
% This function takes in headers from spm_dicom_headers_mh() and checks if
% these headers have the corresponding fields needed for further
% processing. In case these fields are not there the fields are filled. 
% 
% Input: hdr cell array as obtained from spm_dicom_headers_mh
%
% Output : hdr cell array with the same dimensions as the input but
% modified fields in the structures contained in the cell array
%

for i=1:length(hdr)
    
    if ~(isfield(hdr{i},'PixelSpacing')) || ~(isfield(hdr{i},'ImagePositionPatient')) || ~(isfield(hdr{i},'ImageOrientationPatient')) || isfield(hdr{i},'Private_0029_1210'),
    
        hdr{i}.PixelSpacing = hdr{i}.PerFrameFunctionalGroupsSequence{1, 1}.PixelMeasuresSequence{1, 1}.PixelSpacing;  
        hdr{i}.ImagePositionPatient = hdr{i}.PerFrameFunctionalGroupsSequence{1, 1}.PlanePositionSequence{1, 1}.ImagePositionPatient;  
        hdr{i}.ImageOrientationPatient = hdr{i}.PerFrameFunctionalGroupsSequence{1, 1}.PlaneOrientationSequence{1, 1}.ImageOrientationPatient;
    end
end 
disp(['Filled "Image Plane" information for all hdrs.']);
