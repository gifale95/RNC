function hrfs = getcanonicalhrflibrary_glmsingle_spmadapted(len,dt,duration,tr)

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

% inputs
if duration == 0
  duration = 0.1;
end

% load the library
file0 = strrep(which('getcanonicalhrflibrary'),'getcanonicalhrflibrary.m','getcanonicalhrflibrary.tsv');
hrfs = load(file0)';  % 20 HRFs x 501 time points

% convolve to get the predicted response to the desired stimulus duration
trold = 0.1;
hrfs = conv2(hrfs,ones(1,max(1,round(duration/trold))));

% resample to desired TR
hrfs = interp1((0:size(hrfs,2)-1)*trold,hrfs',0:tr:(size(hrfs,2)-1)*trold,'pchip')';  % 20 HRFs x time

% get proportion of canonical hrf that is part of the hrf library
can_hrf_len = len; % length of canonical hrf in seconds
can_hrf_len_tr = round(len/tr); %length of canonical hrf in tr time 
hrfs = hrfs(:,1:can_hrf_len_tr); % take only the spm relevant part 

hrfs = interp1(0:size(hrfs,2)-1,hrfs',0:size(hrfs,2)/(len/dt):size(hrfs,2)-size(hrfs,2)/(len/dt),'pchip')';  % 20 HRFs x time

% make the peak equal to one
hrfs = hrfs ./ repmat(max(hrfs,[],2),[1 size(hrfs,2)]);
end 