function numErr = showRegularizedStokeslets2D(nodes, F, epsilon, nFullGrid, tempk,...
  colorDataLim, PaperPosition, figHandle, stepQuiver, caseProperty, anlzHandle)
% nFullGrid : number of nodes at each dierction.
% tempk: The whole region for velocity field visualization is tempk times larger then
%  the simulation body at each direction.
% colorDataLim: limits of colorbar.
% figHandle: make a dir named by figHandle and short figures whose name
%  began with figHandle.
% stepQuiver: step of Quiver3 graphic function.
% anlzHandle: function handle for analytical solutions for the case.
% Export: 
%          numErr: numerical error(%). 
         
numErr = [];
minLim = mean(nodes)-tempk.*(mean(nodes)-min(nodes));
maxLim = mean(nodes)+tempk.*(max(nodes)-mean(nodes));
fullRegionX = linspace(minLim(1), maxLim(1), nFullGrid(1));
fullRegionY = linspace(minLim(2), maxLim(2), nFullGrid(2));
minLim = mean(nodes) - 1.2*tempk.*(mean(nodes)-min(nodes));
maxLim = mean(nodes) + 1.2*tempk.*(max(nodes)-mean(nodes));
mkdir(figHandle);

[tempX, tempY] = meshgrid(fullRegionX, fullRegionY);
fullNodes = [tempX(:), tempY(:)];
Mfull = RegularizedStokesletsMatrix2D(fullNodes, nodes, epsilon);
Ufull = Mfull * F;
Uxfull = reshape(Ufull(1:2:end),nFullGrid(2),nFullGrid(1));
Uyfull = reshape(Ufull(2:2:end),nFullGrid(2),nFullGrid(1));
Uresultant = reshape(sqrt(Uxfull(:).^2+Uyfull(:).^2),...
  nFullGrid(2),nFullGrid(1));

if isempty(colorDataLim)
  colorDataLim = [min(Uresultant(:)), max(Uresultant(:))];
  fprintf(1, 'New color data limits are set to %5.3e and %5.3e. \n', colorDataLim);
end

fig1 = figure('InvertHardcopy','off','Color',[1 1 1]);
set(fig1, 'Position', get(0,'ScreenSize')+[0 40 0 -120]);
set(fig1, 'visible', 'off');
axes1 = axes('Parent', fig1,...
  'DataAspectRatio', [1 1 1],...
  'box', 'off');
plot(axes1, nodes(:,1),nodes(:,2),...
  'LineWidth', 3,...
  'Color', [0 0 0],...
  'DisplayName', 'geomerty');
hold on;
quiver(tempX(1:stepQuiver:end), tempY(1:stepQuiver:end),...
  Uxfull(1:stepQuiver:end), Uyfull(1:stepQuiver:end),...
  'AutoScaleFactor', 3,...
  'Parent',axes1);
surf(axes1, tempX, tempY, Uresultant,...
  'LineStyle','none',...
  'facealpha', 0.7);
hold off;
set(axes1, 'CLim', colorDataLim,...
  'DataAspectRatio', [1 1 1],...
  'FontSize', 15,...
  'OuterPosition',[0 0 1 1]);
colorbar('peer',axes1);
axis(axes1, 'equal');
xlim(axes1, [minLim(1), maxLim(1)]);
ylim(axes1, [minLim(2), maxLim(2)]);
PNGname = ['.\', figHandle, '\slice.png'];
set(fig1, 'PaperPosition', PaperPosition*get(fig1(1), 'PaperPosition'));
saveas(fig1, PNGname);
close(fig1);

fig1 = figure('InvertHardcopy','off','Color',[1 1 1]);
set(fig1, 'Position', get(0,'ScreenSize')+[0 40 0 -120]);
set(fig1, 'visible', 'off');
axes1 = axes('Parent', fig1,...
  'DataAspectRatio', [1 1 1],...
  'box', 'off');
plot(axes1, nodes(:,1),nodes(:,2),...
  'LineWidth', 3,...
  'Color', [0 0 0],...
  'DisplayName', 'geomerty');
hold on;
contour(axes1, tempX, tempY, Uresultant,...
  'DisplayName', 'numerical result');
if isa(anlzHandle, 'function_handle')
  [UxAnlz, UyAnlz] = anlzHandle(tempX, tempY, caseProperty);
  Uanlz = sqrt(UxAnlz.^2 + UyAnlz.^2);
  contour(axes1, tempX, tempY, Uanlz,...
    'DisplayName', 'analytical results',...
    'LineStyle','--');
  INDEX = ~isnan(Uanlz(:));
  numErr = sqrt(sum((Uresultant(INDEX)-Uanlz(INDEX)).^2) / sum((Uanlz(INDEX)).^2));
end
hold off;
set(axes1, 'CLim', colorDataLim,...
  'DataAspectRatio', [1 1 1],...
  'FontSize', 15,...
  'OuterPosition',[0 0 1 1]);
view(axes1, [0 90]);
axis(axes1, 'equal');
xlim(axes1, [minLim(1), maxLim(1)]);
ylim(axes1, [minLim(2), maxLim(2)]);
legend(axes1,'show');
PNGname = ['.\', figHandle, '\contour.png'];
set(fig1, 'PaperPosition', PaperPosition*get(fig1(1), 'PaperPosition'));
saveas(fig1, PNGname);
close(fig1);

end