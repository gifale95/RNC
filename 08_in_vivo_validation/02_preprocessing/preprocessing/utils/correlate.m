% function r = correlate(X,varargin)
% Function to run correlations, partial correlations, or semi-partial
% correlations.
%
% X is a matrix of values. The output will be the correlation(s).

% TODO: better help for function

function r = correlate(X,varargin)

% set defaults
opt.method = 'corr';
opt.type = 'pearson';

n_var = size(X,2);

% overwrite defaults if required
if nargin > 1
    if ~mod(nargin,2), error('Incorrect number of arguments'), end
    for i = 2:nargin-1
        opt.(varargin{i-1}) = varargin{i};
    end
end

switch lower(opt.method)
    case 'corr'
        r = corr(X,'type',opt.type);
        return
    case 'partialcorr'
        switch lower(opt.type)
            case 'pearson'
                cx = cov(X);
            case 'spearman'
                cx = cov(tiedrank2(X));
            case 'kendall'
                error('Kendall''s tau has not been implemented')
            otherwise
                error('Unknown type %s', opt.type)
        end
        dx = pinv(cx);
        r = -cov2corr(dx);
        r(logical(eye(n_var))) = 1;
    case 'semipartialcorr'
        switch lower(opt.type)
            case 'pearson'
                cx = cov(X);
            case 'spearman'
                cx = cov(tiedrank2(X));
            case 'kendall'
                error('Kendall''s tau has not been implemented')
            otherwise
                error('Unknown type %s', opt.type)
        end
        
        dx = pinv(cx);
        pc = -cov2corr(dx);
        pc(logical(eye(n_var))) = 1;

        dcx = repmat(diag(cx)',n_var,1);
        ddx = repmat(diag(dx)',n_var,1);

        denominator = pc./sqrt(dcx);
        numerator = sqrt(abs(ddx-((dx.^2)'./ddx)'));
        numerator(logical(eye(n_var))) = diag(denominator);
        r = denominator./numerator;
        
    otherwise
        error('unknown method %s input ''method''',opt.method)
end


function r = cov2corr(c)

sdev = diag(diag(c).^(-0.5));
r = sdev*c*sdev;


function allranks = tiedrank2(x)

% faster, less general version of tiedrank (can't deal with inf or NaN)

[sx,sy] = size(x);
r = (1:sx)';

% allranks = zeros(sx,sy); % init
[xsorted,ir] = sort(x);
dxsorted = diff(xsorted,1,1);

for i = sy:-1:1 % invert to save time for preallocation
    f = [0;find(dxsorted(:,i)~=0);sx];
    if length(f)==sx+1 % if all are unique
        allranks(ir(:,i),i) = r;
    else
        for j = length(f)-1:-1:1 % invert to save time for preallocation
            ranks(f(j)+1:f(j+1)) = (f(j)+1+f(j+1))/2; % mean of edges
        end
        allranks(ir(:,i),i) = ranks;
    end
end


%         
% cx = cov([y x1 x2]);
% dx = pinv(cx);
% pc = -cov2corr(dx);
% pc(logical(eye(size(pc)))) = 1;
% 
% a = pc/sqrt(diag(cx)) / sqrt(abs(diag(dx)-((dx^2)'/diag(dx))'));
% 
% dcx = repmat(diag(cx)',3,1);
% ddx = repmat(diag(dx)',3,1);
% 
% denominator = pc./sqrt(dcx);
% numerator = sqrt(abs(ddx-((dx.^2)'./ddx)'));
% numerator(logical(eye(size(numerator)))) = diag(denominator);
% 
% a = denominator./numerator;

% % a = pc./ sqrt(dcx) ./ sqrt(abs(ddx-((dx.^2)'./ddx)'))