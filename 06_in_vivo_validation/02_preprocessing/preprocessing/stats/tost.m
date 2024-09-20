function [p,stat] = tost(test,bound,x1,x2)

% function [p,stat] = tost(test,bound,x1,x2)
% 
% Input:
%   test:
%     'indep_eq': two independent means, equal variances
%     'indep_uneq': two independent means, unequal variances
%     'dep_uneq': two dependent means, unequal variances
%     'one_sample': one sample t-test against mu (statistically equivalent
%       to dependent means, equal variances), when x2 is not provided 0 is
%       assumed
%     'corr': test for lack of association
%   
%   bound: [lower upper] 1 x 2 vector of bound (i.e. equivalence range for mean or correlation)
%   x1: for all options but 'corr', this is data for sample 1, n1 x 1 format 
%       for option 'corr', this is the correlation coefficient (1 x 1 format)
%   x2: for all options with two means, this is sample 2, n2 x 1 format 
%       for option 'one_sample', this is mu that is tested against (1 x 1 format, default: 0)
%       for option 'corr', this is the number of samples going into the correlation coefficient
%
% Output:
%   p: p-value (necessarily two-sided)
%   
%
% Martin Hebart, 2020/09/08

if ~exist('x2','var') || isempty(x2)
    x2 = 0;
end

% bring data in right format if required
if size(x1,1)==1, x1 = x1'; end
if size(x2,1)==1, x2 = x2'; end

n1 = length(x1);
n2 = length(x2);

if numel(bound)~=2
    error('Input variable ''bound'' can only have 2 entries ([lower upper])')
end

lb = bound(1);
ub = bound(2);

if ub < lb
    error('upper bound must be larger than lower bound. Please check input variable ''bound''')
end

switch lower(test)
    
    case 'indep_eq'
        
        M1 = mean(x1);
        M2 = mean(x2);
        sigma = sqrt( 1/(n1+n2-2) * ((n1-1)*var(x1) + (n2-1)*var(x2)) );
        denom = sigma*sqrt(1/n1 + 1/n2);
        tl = (M1-M2-lb)/denom;
        tu = (M1-M2-ub)/denom;
        
        df = n1+n2-2;
        p1 = 1-tcdf(tl,df);
        p2 = tcdf(tu,df);
        p = max(p1,p2);
        if abs(tl)>abs(tu)
            t = tu;
        else
            t = tl;
        end
        
        stat.test = 'independent means, equal variance';
        stat.t = t;
        stat.df = df;
        stat.p = p;
        stat.t_upper = tu;
        stat.t_lower = tl;
        stat.n1 = n1;
        stat.n2 = n2;
        stat.p1 = p1;
        stat.p2 = p2;
        
    case 'indep_uneq'
        
        M1 = mean(x1);
        M2 = mean(x2);        
        
        varx1_norm = var(x1)/n1;
        varx2_norm = var(x2)/n2;
        
        denom = sqrt(varx1_norm + varx2_norm);
        tl = (M1-M2-lb)/denom;
        tu = (M1-M2-ub)/denom;
        
        df = (varx1_norm + varx2_norm)^2 / (varx1_norm^2/(n1-1) + varx2_norm^2/(n2-1));
        p1 = 1-tcdf(tl,df);
        p2 = tcdf(tu,df);
        p = max(p1,p2);
        if abs(tl)>abs(tu)
            t = tu;
        else
            t = tl;
        end
        
        stat.test = 'independent means, unequal variance';
        stat.t = t;
        stat.df = df;
        stat.p = p;
        stat.t_upper = tu;
        stat.t_lower = tl;
        stat.n1 = n1;
        stat.n2 = n2;
        stat.p1 = p1;
        stat.p2 = p2;
        
    case 'dep_uneq'
        
        M1 = mean(x1);
        M2 = mean(x2);  
        varx1 = var(x1);
        varx2 = var(x2);
        r = corr(x1,x2);
        n = n1;
        
        denom = 1/sqrt(n) * sqrt(varx1+varx2 - (2*r*sqrt(varx1)*sqrt(varx2)));
        tl = (M1-M2-lb)/denom;
        tu = (M1-M2-ub)/denom;
        df = n-1;
        
        p1 = 1-tcdf(tl,df);
        p2 = tcdf(tu,df);
        p = max(p1,p2);
        if abs(tl)>abs(tu)
            t = tu;
        else
            t = tl;
        end
        
        stat.test = 'dependent means';
        stat.t = t;
        stat.df = df;
        stat.p = p;
        stat.t_upper = tu;
        stat.t_lower = tl;
        stat.n1 = n1;
        stat.n2 = n2;
        stat.p1 = p1;
        stat.p2 = p2;
        
    case 'one_sample'
        
        M = mean(x1);
        mu = x2;
        n = n1;
        
        denom = std(x1)/sqrt(n);
        tl = (M-mu-lb)/denom;
        tu = (M-mu-ub)/denom;
        df = n-1;

        p1 = 1-tcdf(tl,df);
        p2 = tcdf(tu,df);
        p = max(p1,p2);
        if abs(tl)>abs(tu)
            t = tu;
        else
            t = tl;
        end
        
        stat.test = 'one-sample t test';
        stat.t = t;
        stat.df = df;
        stat.p = p;
        stat.t_upper = tu;
        stat.t_lower = tl;
        stat.n = n;
        stat.p1 = p1;
        stat.p2 = p2;
        
    case 'corr'
        
        r = x1;
        n = x2;
        
        r_norm = atanh(r); % fisher z transform
        lb_norm = atanh(lb);
        ub_norm = atanh(ub);
        
        denom = 1/sqrt(n-3);
        zl = (r_norm - lb_norm)/denom;
        zu = (r_norm - ub_norm)/denom;
        
        p1 = 1-normcdf(zl);
        p2 = normcdf(zu);
        
        p = max(p1,p2);
        if abs(zl)>abs(zu)
            z = zu;
        else
            z = zl;
        end
        
        stat.test = 'correlation coefficient';
        stat.z = z;
        stat.p = p;
        stat.z_upper = zu;
        stat.z_lower = zl;
        stat.n = n;
        stat.p1 = p1;
        stat.p2 = p2;
        
end