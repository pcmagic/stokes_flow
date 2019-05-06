function [T, N, B] = frenetFrame(varargin)
% Inputs:
%          r: curve coordinates;
% Optional inputs:
%          t: time, define is 1:length(r).
% Outputs:
%          [T, N, B]: frenet Frame.
% Zhang Ji, 2016-02-04

init(varargin{:});

T = [diffxy(t,r(1,:)); diffxy(t,r(2,:)); diffxy(t,r(3,:))];
det = sqrt(sum(T.^2,1));
T = [T(1,:)./det; T(2,:)./det; T(3,:)./det];

N = [diffxy(t,r(1,:),[],2); diffxy(t,r(2,:),[],2); diffxy(t,r(3,:),[],2)];
det = sqrt(sum(N.^2,1));
N = [N(1,:)./det; N(2,:)./det; N(3,:)./det];

B = cross(T, N, 1);
det = sqrt(sum(B.^2,1));
B = [B(1,:)./det; B(2,:)./det; B(3,:)./det];

end

function init(varargin)
% This is a general template of parameters initialization for variable
% parameter functions. To avoid user misstakes, the fuction have no output,
% using assignin() function instead. Utlizing this template should:
%  1) edit stdFields and optFields to definite standard and optional
%     parameters, respectively.
%  2) (optionally), Clear up formats of standard parameters. 
%  3) edit optVar to set default values of every optional parameters.
%  4) (optionally), add appropriate rules to avoid illegal parameter values .

stdFields = {'r'};   % standard parameters.
optFields = {'t'};   % optional parameters.
stdNum = length(stdFields);
optNum = length(optFields);

% Get standard parameters.
stdVar = cell2struct(cell(stdNum, 1), stdFields, 1);
try
  for i0 = 1:stdNum
    stdVar.(stdFields{i0}) = varargin{i0};
  end
catch errmsg
  errorMsg = ['Standard parameter ''', stdFields{i0},...
    ''' is missing. Please imput its value. '];
  error('userError:stdValue', errorMsg);
end

% Clear up formats of standard parameters. 
if size(stdVar.r,1) > size(stdVar.r,2)
  stdVar.r = stdVar.r';
end

% Set default values for optional parameters.
% Every optional parameters need default values except it is empty.
optVar = cell2struct(cell(optNum, 1), optFields, 1);
optVar.t = 1:length(stdVar.r);

% Get values of optional parameters.
inoptVar = cell2struct(cell(optNum, 1), optFields, 1);
if nargin == stdNum+1
  if isstruct(varargin{stdNum+1})
    inoptVar = varargin{stdNum+1};
  else
    errorMsg = ['The last parameter should be a struct containing ',...
      'optional parameters and its corresponding values sequentially. '];
    error('userError:optValue', errorMsg);
  end
elseif nargin > stdNum+1
  try
    inoptVar = struct(varargin{stdNum+1:end});
  catch errmsg
    error('userError:optValue', errmsg.message);
  end
end
inoptNames = fieldnames(inoptVar);
for i0 = 1:length(inoptNames)
  if ~any(strcmpi(inoptNames{i0}, optFields))
    error(['Bad option field name: ', inoptNames{i0}])
  end
  if ~isempty(inoptVar.(inoptNames{i0}))
    optVar.(inoptNames{i0}) = inoptVar.(inoptNames{i0});
  end
end

% Assign parameters to father function.
for i0 = 1:stdNum
  assignin('caller', stdFields{i0}, stdVar.(stdFields{i0}));
end
for i0 = 1:optNum
  assignin('caller', optFields{i0}, optVar.(optFields{i0}));
end

% Rules for parameters verification and validation.
if length(stdVar.r) ~= length(optVar.t)
  errorMsg = 'The lengths of curve r and time t must equal. ';
  error('userError:illegalVar', errorMsg);
end

end

function dy = diffxy(x,y,varargin)
% DIFFXY - accurate numerical derivative/differentiation of Y w.r.t X. 
%
%   DY = DIFFXY(X,Y) returns the derivative of Y with respect to X using a 
%        pseudo second-order accurate method. DY has the same size as Y.
%   DY = DIFFXY(X,Y,DIM) returns the derivative along the DIM-th dimension
%        of Y. The default is differentiation along the first 
%        non-singleton dimension of Y.
%   DY = DIFFXY(X,Y,DIM,N) returns the N-th derivative of Y w.r.t. X.
%        The default is 1.
%
%   Y may be an array of any dimension.
%   X can be any of the following:
%       - array X with size(X) equal to size(Y)
%       - vector X with length(X) equal to size(Y,DIM)
%       - scalar X denotes the spacing increment
%   DIM and N are both integers, with 1<=DIM<=ndims(Y)
%
%   DIFFXY has been developed especially to handle unequally spaced data,
%   and features accurate treatment for end-points.
%
%   Example: 
%   % Data with equal spacing
%     x = linspace(-1,2,20);
%     y = exp(x);
% 
%     dy = diffxy(x,y);
%     dy2 = diffxy(x,dy);  % Or, could use >> dy2 = diffxy(x,y,[],2);
%     figure('Color','white')
%     plot(x,(y-dy)./y,'b*',x,(y-dy2)./y,'b^')
%
%     Dy = gradient(y)./gradient(x);
%     Dy2 = gradient(Dy)./gradient(x);
%     hold on
%     plot(x,(y-Dy)./y,'r*',x,(y-Dy2)./y,'r^')
%     title('Relative error in derivative approximation')
%     legend('diffxy: dy/dx','diffxy: d^2y/dx^2',...
%            'gradient: dy/dx','gradient: d^2y/dx^2')
%
%   Example: 
%   % Data with unequal spacing. 
%     x = 3*sort(rand(20,1))-1;
%     % Run the example above from y = exp(x)
%
%   See also DIFF, GRADIENT
%        and DERIVATIVE on the File Exchange

% for Matlab (should work for most versions)
% version 1.0 (Nov 2010)
% (c) Darren Rowland
% email: darrenjrowland@hotmail.com
%
% Keywords: derivative, differentiation

[h,dy,N,perm] = parse_inputs(x,y,varargin);
if isempty(dy)
    return
end
n = size(h,1);
i1 = 1:n-1;
i2 = 2:n;

for iter = 1:N
    v = diff(dy)./h;
    if n>1
        dy(i2,:) = (h(i1,:).*v(i2,:)+h(i2,:).*v(i1,:))./(h(i1,:)+h(i2,:));
        dy(1,:) = 2*v(1,:) - dy(2,:);
        dy(n+1,:) = 2*v(n,:) - dy(n,:);
    else
        dy(1,:) = v(1,:);
        dy(n+1,:) = dy(1,:);
    end
end

% Un-permute the derivative array to match y
dy = ipermute(dy,perm);

%%% Begin local functions %%%
function [h,dy,N,perm] = parse_inputs(x,y,v)

numvarargs = length(v);
if numvarargs > 2
    error('diffxy:TooManyInputs', ...
        'requires at most 2 optional inputs');
end

h = [];
N = [];
perm = [];

% derivative along first non-singleton dimension by default
dim = find(size(y)>1);
% Return if dim is empty
if isempty(dim)
    dy = [];
    return
end
dim = dim(1);

% Set defaults for optional arguments
optargs = {dim 1};
newVals = ~cellfun('isempty', v);
optargs(newVals) = v(newVals);
[dim, N] = optargs{:};

% Error check on inputs
if dim<1 || dim>ndims(y) || dim~=fix(dim) || ~isreal(dim)
    error('diffxy:InvalidOptionalArg',...
        'dim must be specified as a non-negative integer')
end
if N~=fix(N) || ~isreal(N)
    error('diffxy:InvalidOptionalArg',...
        'N must be an integer')
end

% permutation which will bring the target dimension to the front
perm = 1:length(size(y));
perm(dim) = [];
perm = [dim perm];
dy = permute(y,perm);


if length(x)==1  % Scalar expansion to match size of diff(dy,[],1)
    sizeh = size(dy);
    sizeh(1) = sizeh(1) - 1;
    h = repmat(x,sizeh);
elseif ismatrix(x) && any(size(x)==1) % Vector x expansion
    if length(x)~=size(dy,1)
        error('diffxy:MismatchedXandY',...
            'length of vector x must match size(y,dim)')
    end
    x = x(:);
    sizeh = size(dy);
    sizeh(1) = 1;
    h = repmat(diff(x),sizeh);
else
    if size(y) ~= size(x)
        error('diffxy:MismatchedXandY',...
            'mismatched sizes of arrays x and y');
    end
    % Permute x as for y, then diff
    h = diff(permute(x,perm),[],1);
end
end
end