function nodes = HelixDuplicate(varargin)
%--------------------------------------------------------------------------
% Generate a surface mesh of a helix line. 
% Inputs:  
%         r = [r1, r2]: Radius of helix and surface. 
%         w: Angular velocity
%         v: Tangential velocity
%         t: Time
% Optional inputs:
%         perNodes: number of nodes at each layer, define is 16. 
% Outputs: 
%         nodes: mesh information
% Zhang Ji, 2016-03-18
%--------------------------------------------------------------------------

init(varargin{:});

% Obtain the base curve. 
kE = 4 / (v*pi);   % see (BY H. SHUM, E. A. GAFFNEY AND D. J. SMITH, 2010)
f1 = [r(1)*(1-exp(-kE^2*t.^2)).*cos(w*t); r(1)*(1-exp(-kE^2*t.^2)).*sin(w*t); v*t];

% Compute the surface mesh
[T, N, B] = frenetFrame(f1, 't', t);
nodes = ones(3,length(f1)*perNodes);
f2= [r(2)*cos(linspace(0,2*pi,perNodes)); r(2)*sin(linspace(0,2*pi,perNodes)); zeros(1,perNodes)];
for i0 = 1:length(f1)
  JacobiMatrix = [N(:,i0),B(:,i0),T(:,i0)];
  nodes(1:3, (i0-1)*perNodes+1:i0*perNodes) = bsxfun(@plus, JacobiMatrix*f2, f1(:,i0));
end

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

stdFields = {'r', 'w', 'v', 't'};   % standard parameters.
optFields = {'perNodes'};   % optional parameters.
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

% Set default values for optional parameters.
% Every optional parameters need default values except it is empty.
optVar = cell2struct(cell(optNum, 1), optFields, 1);
optVar.perNodes = 16;

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

end