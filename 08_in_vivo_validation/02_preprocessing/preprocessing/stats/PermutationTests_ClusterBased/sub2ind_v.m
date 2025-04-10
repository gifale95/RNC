function ndx = sub2ind_v(siz,varargin)
%SUB2IND Linear index from multiple subscripts.
%   SUB2IND is used to determine the equivalent single index
%   corresponding to a given set of subscript values.
%
%   IND = SUB2IND(SIZ,I,J) returns the linear index equivalent to the
%   row and column subscripts in the arrays I and J for a matrix of
%   size SIZ. 
%
%   IND = SUB2IND(SIZ,I1,I2,...,IN) returns the linear index
%   equivalent to the N subscripts in the arrays I1,I2,...,IN for an
%   array of size SIZ.
%
%   IND = SUB2IND(SIZ,C), where C is a column vector of length R, returns 
%   the linear index equivalent to the subscript of rank R for an array 
%   of size SIZ.
%
%   IND = SUB2IND(SIZ,M), where M is a R x N matrix, returns the N linear 
%   indices equivalent to the N subscripts of rank R in the columns 
%   of M for an array of size SIZ.
%    
%   I1,I2,...,IN must have the same size, and IND will have the same size
%   as I1,I2,...,IN. For an array A, if IND = SUB2IND(SIZE(A),I1,...,IN)),
%   then A(IND(k))=A(I1(k),...,IN(k)) for all k.
%
%   Class support for inputs I,J: 
%      float: double, single
%
%   See also IND2SUB.

%   Copyright 1984-2008 The MathWorks, Inc.
%   Modified portions only:
%       Copyright (c) 2012, Lorenzo Costanzia
%       All rights reserved.
% 
%       Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 
%           Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
%           Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
% 
%       THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
%==============================================================================

siz = double(siz);


d = nargin-1; %number of dimensions (rank) of subs
s = size(varargin{1});

if d == 1 
    %Assume 2nd arg is matrix whose cols are subscripts
    %s(1): num of dimensions, must be equal to length(siz)
    %s(2): num of indices to output
    
    d = s(1);
    m = varargin{1}; %much faster than cell2mat
else
    m = cell2mat(varargin.');
    %will throw error if input vecs have different lengths
    %should ideally throw error(message('MATLAB:sub2ind:SubscriptVectorSize'))
    %but msg by cell2mat is clear enough
end

if length(siz) ~= d
    %Adjust input
    if length(siz)<d
        %Adjust for trailing singleton dimensions
        siz = [siz ones(1,d-length(siz))];
    else
        %Adjust for linear indexing on last element
        siz = [siz(1:d-1) prod(siz(d:end))];
    end
end


%Compute linear indices
k = [1 cumprod(siz(1:end-1))];

%input checking
if any(m(:) < 1) || any(any(m > repmat(siz.', 1, s(2))))
    %Verify subscripts are within range
    error(message('MATLAB:sub2ind:IndexOutOfRange'));
end

ndx = 1 + k * (m-1);

