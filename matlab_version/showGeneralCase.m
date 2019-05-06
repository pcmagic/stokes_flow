function numErr = showGeneralCase(nodes, stokesletHandle, stokesletProperty, F,...
  nFullGrid, tempk, colorDataLim, PaperPosition, figHandle, stepQuiver,...
  geoHandle, caseProperty, anlzHandle)
% nFullGrid : number of nodes at each dierction.
% tempk: The whole region for velocity field visualization is tempk times larger then
%  the simulation body at each direction.
% colorDataLim: limits of colorbar.
% figHandle: make a dir named by figHandle and short figures whose name
%  began with figHandle.
% stepQuiver: step of Quiver3 graphic function.
% geoHandle: function handle for geometries of the case.
% anlzHandle: function handle for analytical solutions for the case.

sumErr2 = nan(1, nFullGrid(3));
sumAnlz2 = nan(1, nFullGrid(3));
minLim = mean(nodes)-tempk.*(mean(nodes)-min(nodes));
maxLim = mean(nodes)+tempk.*(max(nodes)-mean(nodes));
fullRegionX = linspace(minLim(1), maxLim(1), nFullGrid(1));
fullRegionY = linspace(minLim(2), maxLim(2), nFullGrid(2));
fullRegionZ = linspace(maxLim(3), minLim(3), nFullGrid(3));
% fullRegionZ = linspace(maxLim(3), mean(nodes(:,3)), nFullGrid(3));
minLim = mean(nodes) - 1.2*tempk.*(mean(nodes)-min(nodes));
maxLim = mean(nodes) + 1.2*tempk.*(max(nodes)-mean(nodes));

constantColorDataLim = false;
DelayTime = 0.2; 
mkdir(figHandle);

% waitbar1 = waitbar(0,...
%   sprintf('%s, step:%4d/%4d. ',figHandle , 0, nFullGrid(3)),...
%   'Name','Exporting figures');
for i0 = 1:nFullGrid(3)
  if exist('waitbar1', 'var')
    waitbar(i0/nFullGrid(3), waitbar1,...
      sprintf('%s, step:%4d/%4d. ',figHandle , i0, nFullGrid(3)));
  end
  [tempX, tempY, tempZ] = meshgrid(fullRegionX, fullRegionY, fullRegionZ(i0));
  fullNodes = [tempX(:), tempY(:), tempZ(:)];
  Mfull = stokesletHandle(fullNodes, stokesletProperty); 
  Ufull = Mfull * F;
  0 = reshape(Ufull(1:3:end),nFullGrid(2),nFullGrid(1));
  Uyfull = reshape(Ufull(2:3:end),nFullGrid(2),nFullGrid(1));
  Uzfull = reshape(Ufull(3:3:end),nFullGrid(2),nFullGrid(1));
  Uresultant = reshape(sqrt(Uxfull(:).^2+Uyfull(:).^2+Uzfull(:).^2),...
    nFullGrid(2),nFullGrid(1));
  
  if isempty(colorDataLim)
    colorDataLim = [min(Uresultant(:)), max(Uresultant(:))];
    fprintf(1, 'New color data limits are set to %5.3e and %5.3e. \n', colorDataLim);
  else
    constantColorDataLim = true;
  end
  if ~constantColorDataLim &&...
      (min(Uresultant(:))<colorDataLim(1) || max(Uresultant(:))>colorDataLim(2))
    colorDataLim = [min(Uresultant(:)), max(Uresultant(:))];
    fprintf(1, 'New color data limits are set to %5.3e and %5.3e. \n', colorDataLim);
  end
  
  fig1 = figure('InvertHardcopy','off','Color',[1 1 1]);
  set(fig1, 'Position', [1, 1, 768, 1024]);
  set(fig1, 'visible', 'off');
  axes1 = axes('Parent', fig1,...
    'DataAspectRatio', [1 1 1],...
    'box', 'off');
  plot3(axes1, nodes(:,1),nodes(:,2),nodes(:,3));
  hold on;
  quiver3(tempX(1:stepQuiver:end), tempY(1:stepQuiver:end), tempZ(1:stepQuiver:end),...
    Uxfull(1:stepQuiver:end), Uyfull(1:stepQuiver:end), Uzfull(1:stepQuiver:end),...
    'AutoScaleFactor', 3,...
    'Parent',axes1);
  surf(axes1, tempX, tempY, tempZ, Uresultant,...
    'LineStyle','none',...
    'facealpha', 0.7);
  hold off;
  set(axes1, 'CLim', colorDataLim,...
    'DataAspectRatio', [1 1 1],...
    'FontSize', 15,...
    'OuterPosition',[0 0 1 1]);
  colorbar('peer',axes1);
  view(axes1, [-28 58]);
  axis(axes1, 'equal');
  xlim(axes1, [minLim(1), maxLim(1)]);
  ylim(axes1, [minLim(2), maxLim(2)]);
  zlim(axes1, [minLim(3), maxLim(3)]);
  title(axes1, ['Z = ', num2str(fullRegionZ(i0))]);
  set(fig1, 'PaperPosition', PaperPosition*get(fig1(1), 'PaperPosition'));
  drawnow;
  figname = ['.\', figHandle, '\slice_', num2str(i0)];
  saveas(fig1, figname, 'png');
  im = frame2im(getframe(fig1));
  [temp, map] = rgb2ind(im, 10240);
  if i0 == 1;
    imwrite(temp, map, ['.\', figHandle, '\slice.gif'],...
      'LoopCount', Inf,...
      'DelayTime', DelayTime);
  else
    imwrite(temp, map, ['.\', figHandle,  '\slice.gif'],...
      'WriteMode', 'append',...
      'DelayTime', DelayTime);
  end
  close(fig1);
  
%   fig1 = figure('InvertHardcopy','off','Color',[1 1 1]);
%   set(fig1, 'Position', get(0,'ScreenSize')+[0 40 0 -120]);
%   set(fig1, 'visible', 'off');
%   axes1 = axes('Parent', fig1,...
%     'DataAspectRatio', [1 1 1],...
%     'box', 'off');
%   if isa(geoHandle, 'function_handle')
%     [geoX, geoY] = geoHandle(tempX, tempY, tempZ, caseProperty);
%     plot(axes1, geoX, geoY,...
%       'LineWidth', 3,...
%       'Color', [0 0 0],...
%       'DisplayName', 'geomerty');
%   end
%   hold on;
%   contour(axes1, tempX, tempY, Uresultant,...
%     'DisplayName', 'numerical result');
%   if isa(anlzHandle, 'function_handle')
%     [UxAnlz, UyAnlz, UzAnlz] = anlzHandle(tempX, tempY, tempZ, caseProperty);
%     Uanlz = sqrt(UxAnlz.^2 + UyAnlz.^2 + UzAnlz.^2);
%     contour(axes1, tempX, tempY, Uanlz,...
%       'DisplayName', 'analytical results',...
%       'LineStyle','--');
%     INDEX = ~isnan(Uanlz(:));
%     sumErr2(i0) = sum((Uresultant(INDEX)-Uanlz(INDEX)).^2);
%     sumAnlz2(i0) = sum((Uanlz(INDEX)).^2);
%   end
%   hold off;
%   set(axes1, 'CLim', colorDataLim,...
%     'DataAspectRatio', [1 1 1],...
%     'FontSize', 15,...
%     'OuterPosition',[0 0 1 1]);
%   view(axes1, [0 90]);
%   axis(axes1, 'equal');
%   xlim(axes1, [minLim(1), maxLim(1)]);
%   ylim(axes1, [minLim(2), maxLim(2)]);
%   zlim(axes1, [minLim(3), maxLim(3)]);
%   title(axes1, ['Z = ', num2str(fullRegionZ(i0))]);
%   legend(axes1,'show');
%   set(fig1, 'PaperPosition', PaperPosition*get(fig1(1), 'PaperPosition'));
%   drawnow;
%   figname = ['.\', figHandle, '\contour_', num2str(i0)];
%   saveas(fig1, figname, 'png');
%   saveas(fig1, figname, 'fig');
%   im = frame2im(getframe(fig1));
%   [temp, map] = rgb2ind(im, 10240);
%   if i0 == 1;
%     imwrite(temp, map, ['.\', figHandle, '\contour.gif'],...
%       'LoopCount', Inf,...
%       'DelayTime', DelayTime);
%   else
%     imwrite(temp, map, ['.\', figHandle,  '\contour.gif'],...
%       'WriteMode', 'append',...
%       'DelayTime', DelayTime);
%   end
%   close(fig1);
end

numErr = sqrt( sum(sumErr2)/sum(sumAnlz2) );
if exist('waitbar1', 'var')
  delete(waitbar1);
end
end