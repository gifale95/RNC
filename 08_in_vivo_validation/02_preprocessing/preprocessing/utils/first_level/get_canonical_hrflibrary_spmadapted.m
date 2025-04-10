function hrfs = get_canonical_hrflibrary_spmadapted(len,dt)

% function hrfs = getcanonicalhrflibrary(duration,tr)
%
% <duration> is the duration of the stimulus in seconds.
%   should be a multiple of 0.1 (if not, we round to the nearest 0.1).
%   0 is automatically treated as 0.1.
% <tr> is the TR in seconds.
%
% generate a library of 20 predicted HRFs to a stimulus of
% duration <duration>, with data sampled at a TR of <tr>.
%
% the resulting HRFs are returned as 20 x time. the first point is 
% coincident with stimulus onset. each HRF is normalized such 
% that the maximum value is one.
%
% example:
% hrfs = getcanonicalhrflibrary(4,1);
% figure; plot(0:size(hrfs,2)-1,hrfs,'o-');

% load the library
file0 = strrep(which('getcanonicalhrflibrary'),'getcanonicalhrflibrary.m','getcanonicalhrflibrary.tsv');
hrfs = load(file0)';  % 20 HRFs x 501 time points

% Note: we don't need to convolve the hrf with the stimulus function since
% SPM is doing that 
% % convolve to get the predicted response to the desired stimulus duration
trold = 0.1;
% hrfs = conv2(hrfs,ones(1,max(1,round(duration/trold))));

% resample to desired TR
%hrfs = interp1((0:size(hrfs,2)-1)*trold,hrfs',0:tr:(size(hrfs,2)-1)*trold,'pchip')';  % 20 HRFs x time

% get proportion of canonical hrf that is part of the hrf library
can_hrf_len = len; % length of canonical hrf in seconds
hrf_cutoff_len = round(can_hrf_len/trold); % cutoff time in hrf library time
hrfs = hrfs(:,1:hrf_cutoff_len); % take only the spm relevant part 

% interpolate to SPM timeframe
hrfs = interp1(0:size(hrfs,2)-1,hrfs',0:size(hrfs,2)/(len/dt):size(hrfs,2)-size(hrfs,2)/(len/dt),'pchip')';  % 20 HRFs x time

% make the peak equal to one
hrfs = hrfs ./ repmat(max(hrfs,[],2),[1 size(hrfs,2)]);
end 