function mainRegularizedStokeslets2D()

% generate mesh, cylinder case. 
nNode = 20*(2.^(0:5));
eta = 0.01:0.01:0.5;  % eta using at: epsilon = eta(i0)*deltaS;
n_eta = length(eta);
n_nNode = length(nNode);
[eta, nNode] = meshgrid(eta, nNode);
eta = eta(:); nNode = nNode(:);
nCase = length(eta);
numErr = zeros(nCase, 1);

r = 0.25;
u0 = 1;
colorDataLim = [0, 1];
nFullGrid = [53, 53];
tempk = [3, 3];
stepQuiver = 4;
PaperPosition = 0.5;
anlzHandle = @cylinderMotionCaseAnlz;
caseProperty = struct('radius', r,...
  'center', [0,0],...
  'u0', u0);

for i0 = 1:nCase
  deltaS = 2*pi*r/nNode(i0);
  theta = linspace(0,2*pi, nNode(i0));
  nodes = r.*[cos(theta); sin(theta)]';
  
  U = zeros(nNode(i0)*2, 1);
  U(1:2:end) = u0;
  epsilon = eta(i0)*deltaS;
  M = RegularizedStokesletsMatrix2D(nodes, nodes, epsilon);
  F = gmres(M, U);   % Generalized minimal residue method
  figHandle = sprintf('cylinderCase2D_nNode%04d_shiftT%5.3f', nNode(i0), eta(i0));
  numErr(i0) = showRegularizedStokeslets2D(nodes, F, epsilon, nFullGrid, tempk,...
    colorDataLim, PaperPosition, figHandle, stepQuiver, caseProperty, anlzHandle);
end

%--------------------------------------------------------------------------
% show numerical error
eta = reshape(eta, n_nNode, n_eta);
nNode = reshape(nNode, n_nNode, n_eta);
numErr = reshape(numErr, n_nNode, n_eta);

figure;
hold on;
for i0 = 1:size(eta, 1)
  DisplayName = ['N=', num2str(nNode(i0, 1))];
  semilogy(eta(i0, :), numErr(i0,:), 'DisplayName', DisplayName);
end
hold off

log10nNode = log10(nNode);
log10numErr = log10(numErr);
figure;
hold on;
conv = ones(size(nNode, 2), 1);
for i0 = 1:size(nNode, 2)
  DisplayName = ['?=', num2str(eta(1, i0)), '�?s'];
  loglog(nNode(:, i0), numErr(:, i0), 'DisplayName', DisplayName);
  p = polyfit(log10nNode(:, i0), log10numErr(:, i0), 1);
  conv(i0) = p(1);
end
hold off
end

function [UxAnlz, UyAnlz] = cylinderMotionCaseAnlz(tempX, tempY, caseProperty)
% Give the analytical solution of flow past a cylinder.

a = caseProperty.radius;
center = caseProperty.center;
u0 = -caseProperty.u0;

tempX = tempX - center(1);
tempY = tempY - center(2);
r2 = tempX.^2 + tempY.^2;
r = sqrt(r2);
r(r<a) = NaN;
r2(r<a) = NaN;

f0x = 8 * pi/(1-2*log(a)) * u0;
f0y = 0;
temp1 = (2.*log(r)-a^2./r2) ./ (-8*pi);
temp2 = (1-a^2./r2) ./ (4*pi.*r2);
UxAnlz = temp1.*f0x + temp2.*(tempX.*tempX.*f0x + tempX.*tempY.*f0y);
UyAnlz = temp1.*f0y + temp2.*(tempX.*tempY.*f0x + tempY.*tempY.*f0y);

end
