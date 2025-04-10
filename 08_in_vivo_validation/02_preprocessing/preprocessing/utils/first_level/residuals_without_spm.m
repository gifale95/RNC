function residuals = residuals_without_spm(res_names, masks) 

%% load mask(s)

if ~isempty(masks)
    
    settings = decoding_defaults;
    
    if ~isnumeric(masks) 
        settings.files.mask = masks;
        fprintf('     ');
        masks = load_mask(settings);
    end
    
    sz = size(masks);
    mask_index = find(masks);
    maskXYZ = zeros(length(mask_index),3);
    [maskXYZ(:,1),maskXYZ(:,2),maskXYZ(:,3)] = ind2sub(sz,mask_index);
end

%% Load data
if ~exist('Y','var')
    fprintf('     Loading data...\n')
    
    if ~exist('maskXYZ','var')
        error('Neither passed data nor masks. Unclear which voxels to load. Pass at least masks')
    end
    
    residuals = zeros(length(res_names),length(mask_index));
    for i_vol = 1:length(res_names)
        residuals(i_vol,:) = read_voxels(settings.software,spm_vol(res_names{i_vol}),maskXYZ);
    end
end
end 