function sub = ind2sub1(siz, ind)
%
%   Created 2007-03-26   Paul A.M. Bune, Alcatel-Lucent Deutschland AG
%   LastMod 2008-04-08   Paul A.M. Bune, Alcatel-Lucent Deutschland AG
%
% sub = ind2sub1(siz, ind)
%
%   Multiple subscripts (written into one vector or matrix) from linear
%   index "ind" for an N-dimensional array defined by its array size "siz".
%
%   Linear index "ind" can be a single positive integer or a vector of
%   positive integers. If "ind" is a row vector, it is converted into a
%   column vector internally.
%
%   In principle, the same as MatLab routine "ind2sub", but the result is
%   put into an automatically sized vector (if "ind" is a single integer)
%   or matrix (if "ind" is a vector) rather than into multiple variables.
%
%   Example:
%
%     A    = ones(2, 3, 4);
%     Sub1 = ind2sub1(size(A), 6)
%
%     Sub1 =
%
%          2     3     1
%
%     Sub2 = ind2sub1(size(A), 4:4:24)
%
%     Sub2 =
%          2     2     1
%          2     1     2
%          2     3     2
%          2     2     3
%          2     1     4
%          2     3     4
%
%

if numel(ind) == 1

  if mod(ind, 1) ~= 0 || ind < 1
    error('"ind" has to be a positive integer');
  end

  if ind > prod(siz)
    error('"ind" exceeds the number of elements defined by "siz"');
  end

  sub = 1 + mod(ceil(repmat(ind, 1, numel(siz)) ./ ...
    cumprod([1 siz(1:(end-1))])) - 1, siz);

else

  if length(size(ind)) > 2 || min(size(ind)) > 1
    error('"ind" has to be a scalar or vector');
  end

  if ~isempty(find(mod(ind, 1), 1)) || min(ind) < 1
    error('"ind" shall only contain positive integers');
  end

  if max(ind) > prod(siz)
    error(['At least one element in "ind" exceeds the number of ' ...
      'elements defined by "siz"']);
  end

  sub = 1 + mod(ceil(repmat(ind(:), 1, numel(siz)) ./ ...
    repmat(cumprod([1 siz(1:(end-1))]), numel(ind), 1)) - 1, ...
    repmat(siz, numel(ind), 1));

end
return