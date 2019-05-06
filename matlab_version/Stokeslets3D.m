function Stokeslets3D(nodes, gNodes, U, nFullGrid, tempk, colorDataLim, figHandle, stepQuiver)
% Solve rigid body locomote in Stokes flow using ghost node stokeslet method.
% Force and velocity information are storaged in different (but one-one
%  correspondance) nodes.
% Originated form Prof. Xu xinliang in 2016-03-11
% Motivated by Zhang Ji in 2016-03-18
% Inputs: 
%           nodes: node information;
%           gNodes: ghost node information;
%           U: initial conditions, velocity.
%           nFullGrid : number of nodes at each dierction.
%           tempk: The whole region for velocity field visualization is 
%                   tempk times larger then the simulation body at each
%                   direction.
%           colorDataLim: limits of colorbar.
%           figHandle: make a dir named by figHandle and short figures 
%                       whose name began with figHandle.
%           stepQuiver: step of Quiver3 graphic function.

fig1 = figure('Position', get(0,'ScreenSize')+[0 40 0 -120]);
plot3(nodes(:,1), nodes(:,2), nodes(:,3), 'r.', 'DisplayName', 'rotated nodes');
hold on;
plot3(gNodes(:,1), gNodes(:,2), gNodes(:,3), 'g.','DisplayName', 'ghost nodes');
quiver3(nodes(:,1), nodes(:,2), nodes(:,3),...
  U(1:3:end), U(2:3:end), U(3:3:end),...
  'AutoScaleFactor', 1,...
  'DisplayName','velocity');
hold off;
axis equal;
legend(gca,'show');

% Solve forces on the boundary
M = stokesletMatrix(nodes, gNodes);
nNode = length(nodes(:));
F = gmres(M, U, nNode-50, 1e-6);   % Generalized minimal residue method
% F2 = M\U;

% Post process
% RU = M * F;
showGeneralCase(nodes, gNodes, F, nFullGrid, tempk, colorDataLim, figHandle, stepQuiver);
close(fig1); clear fig1;
save(['.\', figHandle, '\', figHandle, '.mat']);
end

function M = stokesletMatrix(nodes, gNodes)
% Solve M matrix
% Input:
%          nodes: velocity U node location infromation;
%          gNodes: force F node location information.
% Output:
%          M: U = M * F.
nNode   = length(nodes);
ngNodet = length(gNodes);
M = zeros(nNode*3, ngNodet*3);
for i0 = 1:nNode
  deltaXi = bsxfun(@minus, gNodes, nodes(i0,:));
  deltaR2 = sum(deltaXi.^2, 2);
  deltaR = sqrt(deltaR2);
  temp1 = 1 ./ (8*pi*deltaR);
  temp2 = 1 ./ (8*pi*deltaR.*deltaR2);
  M(3*i0-2, 1:3:end) = temp1 + deltaXi(:,1).*deltaXi(:,1).*temp2;   % Mxx
  M(3*i0-1, 2:3:end) = temp1 + deltaXi(:,2).*deltaXi(:,2).*temp2;   % Myy
  M(3*i0,   3:3:end) = temp1 + deltaXi(:,3).*deltaXi(:,3).*temp2;   % Mzz
  M(3*i0-2, 2:3:end) =         deltaXi(:,1).*deltaXi(:,2).*temp2;   % Mxy
  M(3*i0-2, 3:3:end) =         deltaXi(:,1).*deltaXi(:,3).*temp2;   % Mxz
  M(3*i0-1, 3:3:end) =         deltaXi(:,2).*deltaXi(:,3).*temp2;   % Myz
  M(3*i0-1, 1:3:end) = M(3*i0-2, 2:3:end);                          % Myx
  M(3*i0,   1:3:end) = M(3*i0-2, 3:3:end);                          % Mzx
  M(3*i0,   2:3:end) = M(3*i0-1, 3:3:end);                          % Mzy
end
end

function showGeneralCase(nodes, gNodes, F, nFullGrid, tempk, colorDataLim, figHandle, stepQuiver)
% nFullGrid : number of nodes at each dierction.
% tempk: The whole region for velocity field visualization is tempk times larger then
%  the simulation body at each direction.
% colorDataLim: limits of colorbar.
% figHandle: make a dir named by figHandle and short figures whose name
%  began with figHandle.
% stepQuiver: step of Quiver3 graphic function.

minLim = mean(nodes)-tempk.*(mean(nodes)-min(nodes));
maxLim = mean(nodes)+tempk.*(max(nodes)-mean(nodes));
fullRegionX = linspace(minLim(1), maxLim(1), nFullGrid(1));
fullRegionY = linspace(minLim(2), maxLim(2), nFullGrid(2));
fullRegionZ = linspace(maxLim(3), mean(nodes(:,3)), nFullGrid(3));
minLim = mean(nodes) - 1.2*tempk.*(mean(nodes)-min(nodes));
maxLim = mean(nodes) + 1.2*tempk.*(max(nodes)-mean(nodes));
constantColorDataLim = false;
mkdir(figHandle);

waitbar1 = waitbar(0, sprintf('step:%4d%4d', 0, nFullGrid(3)),...
  'Name','Exporting figures of slices');
for i0 = 1:nFullGrid(3)
  waitbar(i0/nFullGrid(3), waitbar1,...
    sprintf('%s, step:%4d/%4d. ',figHandle , i0, nFullGrid(3)));
  [tempX, tempY, tempZ] = meshgrid(fullRegionX, fullRegionY, fullRegionZ(i0));
  fullNodes = [tempX(:), tempY(:), tempZ(:)];
  Mfull = stokesletMatrix(fullNodes, gNodes);
  Ufull = Mfull * F;
  Uxfull = reshape(Ufull(1:3:end),nFullGrid(2),nFullGrid(1));
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
  set(fig1, 'Position', get(0,'ScreenSize')+[0 40 0 -120]);
  set(fig1, 'visible', 'off');
  axes1 = axes('Parent', fig1,...
    'DataAspectRatio', [1 1 1],...
    'box', 'off');
  plot3(axes1, nodes(:,1),nodes(:,2),nodes(:,3));
  hold on;
  quiver3(tempX(1:stepQuiver:end), tempY(1:stepQuiver:end), tempZ(1:stepQuiver:end),...
    Uxfull(1:stepQuiver:end), Uyfull(1:stepQuiver:end), Uzfull(1:stepQuiver:end),...
    'AutoScaleFactor', 3,...
    'Color',[0.800000011920929 0.800000011920929 0.800000011920929],...
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
  PNGname = ['.\', figHandle, '\', num2str(i0), '.png'];
  set(fig1, 'PaperPosition', round(stepQuiver/2)*get(fig1(1), 'PaperPosition'));
  saveas(fig1, PNGname);
  close(fig1);
end
delete(waitbar1);

end