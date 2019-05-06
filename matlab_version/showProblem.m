function fig1 = showProblem(varargin)

nodes = [];       % nodes of the problem.
gNodes = [];      % ghost nodes of the problem, Ding's method.
U = [];           % velocity of the problem.
init(varargin{:});

fig1 = figure('Position', get(0,'ScreenSize')+[0 40 0 -120]);
plot3(nodes(:,1), nodes(:,2), nodes(:,3), 'r.', 'DisplayName', 'rotated nodes');
hold on;
if ~isempty(gNodes)
  plot3(gNodes(:,1), gNodes(:,2), gNodes(:,3), 'g.','DisplayName', 'ghost nodes');
end
if ~isempty(U)
  quiver3(nodes(:,1), nodes(:,2), nodes(:,3),...
    U(1:3:end), U(2:3:end), U(3:3:end),...
    'DisplayName','velocity');
end
hold off;
axis equal;
legend(gca,'show');
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

stdFields = {'nodes'};   % standard parameters.
optFields = {'gNodes', 'U'};   % optional parameters.
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
num_node = size(stdVar.nodes);
if num_node(1) < num_node(2)
  stdVar.nodes = stdVar.nodes';
end

% Set default values for optional parameters.
% Every optional parameters need default values except it is empty.
optVar = cell2struct(cell(optNum, 1), optFields, 1);
optVar.gNodes = [];
optVar.U = [];

% Get values of optional parameters.
% inoptVar = cell2struct(cell(optNum, 1), optFields, 1);
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

% Clear up formats of optional parameters.
num_gnode = size(optVar.gNodes);
if num_gnode(1) < num_gnode(2)
  optVar.gNodes = optVar.gNodes';
end

% Rules for parameters verification and validation.
if ~isempty(optVar.gNodes)
  if length(stdVar.nodes) ~= length(optVar.gNodes)
    errorMsg = 'numbers of node and ghost node must equal. ';
    error('userError:inputValue', errorMsg);
  end
  if (size(stdVar.nodes, 2) ~= 3) || (size(optVar.gNodes, 2) ~= 3)
    errorMsg = 'only 3D case supported. ';
    error('userError:inputValue', errorMsg);
  end
end
if ~isempty(optVar.U)
  if length(stdVar.nodes)*3 ~= length(optVar.U)
    errorMsg = 'only 3D case supported. ';
    error('userError:inputValue', errorMsg);
  end
end

% Assign parameters to father function.
for i0 = 1:stdNum
  assignin('caller', stdFields{i0}, stdVar.(stdFields{i0}));
end
for i0 = 1:optNum
  assignin('caller', optFields{i0}, optVar.(optFields{i0}));
end
end