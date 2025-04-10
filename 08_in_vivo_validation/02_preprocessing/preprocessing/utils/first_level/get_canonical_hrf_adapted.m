function hrfs = get_canonical_hrf_adapted(dt,len)

% load the library
file0 = strrep(which('getcanonicalhrflibrary'),'getcanonicalhrflibrary.m','getcanonicalhrflibrary.tsv');
hrfs = load(file0)';  % 20 HRFs x 501 time points

% % convolve to get the predicted response to the desired stimulus duration
% trold = 0.1;
% hrfs = conv2(hrfs,ones(1,max(1,round(duration/trold))));

% resample to desired TR
hrfs = interp1(0:size(hrfs,2)-1,hrfs',0:size(hrfs,2)/(len/dt):size(hrfs,2)-size(hrfs,2)/(len/dt),'pchip')';  % 20 HRFs x time

% make the peak equal to one
hrfs = hrfs ./ repmat(max(hrfs,[],2),[1 size(hrfs,2)]);

end 