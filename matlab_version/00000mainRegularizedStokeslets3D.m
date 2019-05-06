function mainRegularizedStokeslets3D()

%--------------------------------------------------------------------------
% Helix locomotion case
colorDataLim = [3e-6, 12e-6];
nFullGrid = [53, 53, 11];
tempk = [3, 3, 1];
stepQuiver = 4;
w0 = 1*pi;
v0 = linspace(0, 0.001, 11);
PaperPosition = 1;

nCase = length(v0);
for i0 = 1:1
  figHandle = sprintf('helixCase_shiftT0.95_v%3.2e', v0(i0));
  [nodes, gNodes, U] = helixLocomotionCase(w0, v0(i0));
%   fig1 = showProblem(nodes, gNodes, U);
  M = stokesletMatrix3D(nodes, gNodes);
  [F, flag(i0), relres(i0), iter(i0)] = tfqmr(M, U, [1e-6], length(U)*100);   % Transpose-free quasi-minimal residual method
  showGeneralCase(nodes, gNodes, F,...
    nFullGrid, tempk, colorDataLim, PaperPosition, figHandle, stepQuiver,...
    [], [], []);
  if exist('fig1', 'var')
    close(fig1); clear fig1;
  end
  save(['.\', figHandle, '\', figHandle, '.mat']);
end

end