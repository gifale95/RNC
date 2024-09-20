% function [maskvol,XYZindex] = remove_eyes(k,K,m_thresh,maskname,vollist,testmode)
%
% This function selects the eye balls from a user-defined mask, increases
% the mask size and creates an eye mask. This may lead to better 
% realignment results, because large signal variations from the eye balls 
% are abolished (see e.g. Freire & Mangin (2001) - Neuroimage).
%
% The inverse of this mask may be written and used as a weight map for 
% realignment (implicit masking, default name: 'emask.img'). Alternatively,
% if a list of volumes is provided, the eyes may be removed from these 
% volumes (explicit masking). Newly written files will be prepended with 
% the letter 'e'.
%
% This routine needs user interaction, therefore there is a test mode in
% which the user can check if only the eyes are masked.
%
% (c) Martin Hebart, 2010/04/17
%
% Input variables:
% k: smallest allowed size of each eye ball (for 3x3x3 50, for 2x2x2 100 is
%    a good value)
% K: largest allowed size of each eye ball (1000 seems to work well)
% m_thresh: masking threshold (normal is 0.7, can also go up to 0.9)
% maskname: Name of mask
% vollist: list of files or 0 if no files are to be written
% testmode (optional): if nonzero, the mask will be plotted to check its
%    location
%
% Other ways to tweak the function and how the function works are described
% in the function itself.

% TODO: narrow down search space

function [maskvol,XYZindex] = remove_eyes(k,K,m_thresh,maskname,vollist,testmode)

%%%%%%%%%%%%%%%%%%%%%%%%%%
% How the function works %
%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1. Load image and if necessary turn to mask
% 2. Erode to get rid of skull and to disconnect eye from brain
% 3. Find all clusters within given range
% 4. Dilate those clusters
% 5. Mask the clusters

%%%%%%%%%%%%%%%%%%%%
% Further Tweaking %
%%%%%%%%%%%%%%%%%%%%
% Play around with the number of iteration of spm_erode and spm_dilate
% If eyes can't be found, change n_erode; if eyes are wrong size, change
% n_dilate. Make sure the difference between them stays similar. If all
% this doesn't work, then load an image and increase the masking threshold
% (e.g. from 0.7 to 0.75)
n_erode = 1;
n_dilate = 2;
% m_thresh = 0.9;

% Fill missing variables
if ~exist('vollist','var'), vollist = 0; end
if ~exist('testmode','var'), testmode = 0; end

% Check for good input
if testmode ~= 0 && vollist ~= 0
    error('Testmode was on, but vollist wasn''t 0. Change one of these variables!')
end

% Get volume headers
if vollist ~= 0
    volhdr = spm_vol(vollist);
end

% Get masking volume
maskhdr = spm_vol(maskname);
maskvol = spm_read_vols(maskhdr);
sz = size(maskvol);

% If this is not a mask, but an image
if any(maskvol(:)>1)
    % Turn into mask
    maskvol = double( maskvol > m_thresh * mean(maskvol(:)) );
end

% Erode image to disconnect eyes from brain
for i = 1:n_erode
    maskvol_e = spm_erode(maskvol);
end

% Get voxel indices / coordinates
Q = find(maskvol_e==1);
clear XYZ % just in case
[XYZ(1,:) XYZ(2,:) XYZ(3,:)] = ind2sub(sz,Q);

% run spm_clusters and get clusters with specific size
A     = spm_clusters(XYZ);
Q     = [];

for i = 1:max(A)
j = find(A == i);
if (length(j) >= k && length(j) < K); Q = [Q j]; end %#ok<AGROW>
end

% Update coordinates
XYZ = XYZ(:,Q);

% Use spm_dilate to increase eyes back
ima = zeros(sz);
XYZindex = sub2ind(sz,XYZ(1,:),XYZ(2,:),XYZ(3,:));
ima(XYZindex) = 1;
for i = 1:n_dilate % three iterations
ima = spm_dilate(ima);
end
XYZindex = find(ima);

% XYZindex = grow_roi(XYZ,sz,3); % (alternative function to grow eyes)

% Do rest as specified

% If testmode, plot results
if testmode ~= 0 %#ok<ALIGN>
    maskvol(maskvol>0) = 0.3;
    maskvol(XYZindex) = 1;
    figure
    imagesc(transform_vol(maskvol))
    colormap('gray')
    axis('off','equal')
% Otherwise write inverted mask for realignment
else
    maskvol = ones(sz);
    maskvol(XYZindex) = 0;
    [path,fname,ext] = fileparts(maskhdr.fname);
    maskhdr.fname = fullfile(path,['e' fname ext]);
    spm_write_vol(maskhdr,maskvol);
end

% If volumes are provided, remove eyes in these volumes
if vollist ~= 0
    for ivol = 1:size(vollist,1)    
        currvol = spm_read_vols(volhdr(ivol));
        currvol(XYZindex) = 0;
        [path,fname,ext] = fileparts(volhdr(ivol).fname);
        volhdr(ivol).fname = fullfile(path,['e' fname ext]);
        spm_write_vol(volhdr(ivol),currvol);
    end
end


%----------------------
function newvol = transform_vol(vol)

sz = size(vol);

factor = sz(2)/sz(1);
nRows = round(sqrt(sz(3))*factor);
nColumns = ceil(sz(3)/nRows);

newvol = zeros(sz(1)*nRows,sz(2)*nColumns);

counterRow = 0;
counterColumn = 1;

slicetemplate = newvol;
slicetemplate(1:numel(slicetemplate)) = 1:numel(slicetemplate);
sliceindices = slicetemplate(1:sz(1),1:sz(2))-1;

for i = 1:sz(3)
    if counterRow == nRows
        counterRow = 0;
        counterColumn = counterColumn + 1;
    end
    counterRow = counterRow + 1;
    currindices = sliceindices+slicetemplate((counterRow-1)*sz(1)+1,(counterColumn-1)*sz(2)+1);
    newvol(currindices) = vol(:,:,i);
end