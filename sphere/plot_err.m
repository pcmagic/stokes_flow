function [fig1, gifHeadle] = plot_err(varargin)

init(varargin{:})
shape = size(para1);

fig1 = figure('Color',[1 1 1]);
set(fig1, 'Position', figPosition);
set(fig1, 'visible', 'off');
axes1 = axes('Parent',fig1);
% plot a temporary line and remove it later to set x and y scales.
line0 = plotHeadle(para1(:, 1), err(:, 1),'Parent',axes1);
hold(axes1,'on');
for i0 = 1:shape(2)
    plotHeadle(para1(:, i0), err(:, i0),...
        'Parent',axes1,...
        'DisplayName', [para2Name, ': ', num2str(para2(1, i0))],...
        'marker', '.', 'markersize', 20)
end
delete(line0)
xlabel(axes1, para1Name, 'interpreter', 'none');
ylabel(axes1, yLabel, 'interpreter', 'none');
xlim(axes1,para1Lim);
ylim(axes1,yLim);
title(axes1, Title, 'interpreter', 'none')
box(axes1,'on');
set(axes1,'XGrid','on','XMinorGrid','on','XMinorTick','on',...
    'YGrid','on','YMinorGrid','on','YMinorTick','on');
legend(axes1,'show', 'Location', 'eastoutside');
hold off
drawnow;

figName = ['./', folderHeadle, '/', para1Name, '_', figHandle];
gifHeadle = ['./', folderHeadle, '/', para1Name, '_'];
saveas(fig1, [figName, '.png'], 'png');
saveas(fig1, [figName, '.fig'], 'fig');
set(axes1, 'fontSize', get(axes1, 'fontSize') * 2);
end

% function setScale(axes1, plotHeadle)
% a = 1;
%
% end

function init(varargin)
% This is a general template of parameters initialization for variable
% parameter functions. To avoid user misstakes, the fuction have no output,
% using assignin() function instead. Utlizing this template should:
%  1) edit stdFields and optFields to definite standard and optional
%     parameters, respectively.
%  2) (optionally), Clear up formats of standard parameters.
%  3) edit optVar to set default values of every optional parameters.
%  4) (optionally), add appropriate rules to avoid illegal parameter values .

stdFields = {'para1', 'para2', 'err'};   % standard parameters.
optFields = {'para1Name', 'para2Name', 'yLabel',...   % optional parameters.
    'plotHeadle', 'para1Lim', 'para2Lim', 'yLim',...
    'folderHeadle', 'figHandle', 'Title', 'figPosition', 'DelayTime'};
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
optVar.para1Name = '';
optVar.para2Name = '';
optVar.yLable = '';
optVar.plotHeadle = @plot;
optVar.para1Lim = 'auto';
optVar.para2Lim = 'auto';
optVar.yLim = 'auto';
optVar.folderHeadle = '';
optVar.figHandle = '';
optVar.Title = '';
optVar.figPosition = get(0,'ScreenSize');
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
if sum(stdVar.para1(:, 1) ~= stdVar.para1(:, 2)) && sum(stdVar.para2(:, 1) == stdVar.para2(:, 2))
    temp = stdVar.para1;
    stdVar.para1 = stdVar.para2;
    stdVar.para2 = temp;
    temp = optVar.para1Lim;
    optVar.para1Lim = optVar.para2Lim;
    optVar.para2Lim = temp;
    temp = optVar.para1Name;
    optVar.para1Name = optVar.para2Name;
    optVar.para2Name = temp;
end

% Assign parameters to father function.
for i0 = 1:stdNum
    assignin('caller', stdFields{i0}, stdVar.(stdFields{i0}));
end
for i0 = 1:optNum
    assignin('caller', optFields{i0}, optVar.(optFields{i0}));
end

end