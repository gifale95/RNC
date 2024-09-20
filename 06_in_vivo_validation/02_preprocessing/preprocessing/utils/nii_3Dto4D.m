%% FUNCTION FOR CREATING 4-D NIFTI FILES

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% MODIFIED FROM spm_config_3Dto4D.m (20 june 2007)
%
% Copyright (C) 2005 Wellcome Department of Imaging Neuroscience
%
% John Ashburner
% $Id: spm_config_3Dto4D.m 245 2005-09-27 14:16:41Z guillaume $
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function nii_3Dto4D(files,save_name)

V = spm_vol(strvcat(files{:}));
ind  = cat(1,V.n);
N    = cat(1,V.private);

mx   = -Inf;
mn   = Inf;
for i=1:numel(V),
    dat      = V(i).private.dat(:,:,:,ind(i,1),ind(i,2));
    dat      = dat(isfinite(dat));
    mx       = max(mx,max(dat(:)));
    mn       = min(mn,min(dat(:)));
end

sf         = max(mx,-mn)/32767;
ni         = nifti;
ni.dat     = file_array(sprintf('%s',save_name),[V(1).dim numel(V)],'INT16-BE',0,sf,0);
ni.mat     = N(1).mat;
ni.mat0    = N(1).mat;
ni.descrip = '4D image';
create(ni);
for i=1:size(ni.dat,4),
    ni.dat(:,:,:,i) = N(i).dat(:,:,:,ind(i,1),ind(i,2));
    spm_get_space([ni.dat.fname ',' num2str(i)], V(i).mat);
end

end