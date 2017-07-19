function numErr = checkError(nodes, gNodes, F,...
  nFullGrid, tempk, caseProperty, anlzHandle)
% nFullGrid : number of nodes at each dierction.
% tempk: The whole region for velocity field visualization is tempk times larger then
%  the simulation body at each direction.
% figHandle: make a dir named by figHandle and short figures whose name
%  began with figHandle.
% anlzHandle: function handle for analytical solutions for the case.

sumErr2 = nan(1, nFullGrid(3));
sumAnlz2 = nan(1, nFullGrid(3));
minLim = mean(nodes)-tempk.*(mean(nodes)-min(nodes));
maxLim = mean(nodes)+tempk.*(max(nodes)-mean(nodes));
fullRegionX = linspace(minLim(1), maxLim(1), nFullGrid(1));
fullRegionY = linspace(minLim(2), maxLim(2), nFullGrid(2));
fullRegionZ = linspace(maxLim(3), mean(nodes(:,3)), nFullGrid(3));

for i0 = 1:nFullGrid(3)
  [tempX, tempY, tempZ] = meshgrid(fullRegionX, fullRegionY, fullRegionZ(i0));
  fullNodes = [tempX(:), tempY(:), tempZ(:)];
  Mfull = stokesletMatrix3D(fullNodes, gNodes);
  Ufull = Mfull * F;
  Uxfull = reshape(Ufull(1:3:end),nFullGrid(2),nFullGrid(1));
  Uyfull = reshape(Ufull(2:3:end),nFullGrid(2),nFullGrid(1));
  Uzfull = reshape(Ufull(3:3:end),nFullGrid(2),nFullGrid(1));
  Uresultant = reshape(sqrt(Uxfull(:).^2+Uyfull(:).^2+Uzfull(:).^2),...
    nFullGrid(2),nFullGrid(1));
  
  [UxAnlz, UyAnlz, UzAnlz] = anlzHandle(tempX, tempY, tempZ, caseProperty);
  Uanlz = sqrt(UxAnlz.^2 + UyAnlz.^2 + UzAnlz.^2);
  INDEX = ~isnan(Uanlz(:));
  sumErr2(i0) = sum((Uresultant(INDEX)-Uanlz(INDEX)).^2);
  sumAnlz2(i0) = sum((Uanlz(INDEX)).^2);
end

numErr = sqrt( sum(sumErr2)/sum(sumAnlz2) );

end