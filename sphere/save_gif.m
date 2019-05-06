function save_gif(varargin)

init(varargin{:})
im = frame2im(getframe(figHeadle));
[temp, map] = rgb2ind(im, 256);
if i0 == 1;
    imwrite(temp, map, [figName, '.gif'], 'gif',...
        'LoopCount', Inf,...
        'DelayTime', DelayTime);
else
    imwrite(temp, map, [figName, '.gif'], 'gif',...
        'WriteMode', 'append',...
        'DelayTime', DelayTime);
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

stdFields = {'figHeadle', 'figName'};   % standard parameters.
optFields = {'i0', 'DelayTime'};   % optional parameters.

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
optVar.i0 = 1;
optVar.DelayTime = 0.2;

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
        warning(['Bad option field name: ', inoptNames{i0}])
    end
    if ~isempty(inoptVar.(inoptNames{i0}))
        optVar.(inoptNames{i0}) = inoptVar.(inoptNames{i0});
    end
end

% Rules for parameters verification and validation.


% Assign parameters to father function.
for i0 = 1:stdNum
    assignin('caller', stdFields{i0}, stdVar.(stdFields{i0}));
end
for i0 = 1:optNum
    assignin('caller', optFields{i0}, optVar.(optFields{i0}));
end

end