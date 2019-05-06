function mainStokeslets3D()
% Per process mesh and boundary conditions for Stokeslets3D.m

%-------------------------------------------------------------------------
shiftT = 0.8:0.01:0.99;
nTracer = 50*(2.^(0:4));
n_shiftT = length(shiftT);
n_nTracer = length(nTracer);
[shiftT, nTracer] = meshgrid(shiftT, nTracer);
shiftT =  shiftT(:); nTracer = nTracer(:);
nCase = length(shiftT);
numErr = zeros(nCase, 1);
flag = zeros(nCase, 1);        % properties for GMRES iteration.
relres = zeros(nCase, 1);
iter = zeros(nCase, 1);
iterGMRES = zeros(nCase, 1);
figHandle = 'sphereCase_nTracer_shiftT_tfqmr-length(U)';

colorDataLim = [0, 1];
nFullGrid = [530, 530, 270];
tempk = [3, 3, 3];
stepQuiver = 4;
PaperPosition = 0.5;
anlzHandle = @sphereMotionCaseAnlz;
geoHandle = @sphereMotionCaseGeometry;

% if isempty(gcp('nocreate'))
%   parpool(24);
% end
% pctRunOnAll javaaddpath java;
% ppm = ParforProgMon('Sphere Case: ', nCase, 1, 300, 80);
for i0 = 1:nCase
%   ppm.increment();

  %   fprintf(1, 'sphereCase_shiftT%.2e', shiftT(i0));
  [nodes, gNodes, U, caseProperty] = sphereMotionCase(shiftT(i0), nTracer(i0));
%     fig1 = showProblem(nodes, 'gNodes', gNodes, 'U', U);
  M = stokesletMatrix3D(nodes, gNodes);
%   [F, flag(i0), relres(i0), iterGMRES(i0,:)] = gmres(M, U, 100, 1e-4, 100);   % Generalized minimal residue method
  [F, flag(i0), relres(i0), iter(i0)] = tfqmr(M, U, 1e-6, length(U)*100);   % Transpose-free quasi-minimal residual method
%     numErr(i0) = showGeneralCase(nodes, gNodes, F,...
%       nFullGrid, tempk, colorDataLim, PaperPosition, figHandle, stepQuiver,...
%       geoHandle, caseProperty, anlzHandle);
  numErr(i0) = checkError(nodes, gNodes, F, nFullGrid, tempk, caseProperty, anlzHandle);
%     if exist('fig1', 'var')
%       close(fig1); clear fig1;
%     end
%     save(['.\', figHandle, '\', figHandle, '.mat']);
end
% ppm.delete()
save(figHandle);

%--------------------------------------------------------------------------
% shiftT = 0.1:0.05:0.9;
% nTracer = 50*(2.^(0:6));
% n_shiftT = length(shiftT);
% n_nTracer = length(nTracer);
% [shiftT, nTracer] = meshgrid(shiftT, nTracer);
% shiftT =  shiftT(:); nTracer = nTracer(:);
% nCase = length(shiftT);
% numErr = zeros(nCase, 1);
% flag = zeros(nCase, 1);        % properties for GMRES iteration.
% relres = zeros(nCase, 1);
% iter = zeros(nCase, 1);
% % iterGMRES = zeros(nCase, 1);
% figHandle = 'sphereCase_nTracer_shiftT_tfqmr-length(U)';
% colorDataLim = [0, 1];
% nFullGrid = [53, 53, 27];
% tempk = [3, 3, 3];
% stepQuiver = 4;
% PaperPosition = 0.5;
%
% [nodes, gNodes, U, caseProperty] = sphereMotionCase(shiftT(i0), nTracer(i0));
% fig1 = showProblem(nodes, 'gNodes', gNodes, 'U', U);
% M = stokesletMatrix3D(nodes, gNodes);
% %   [F, flag(i0), relres(i0), iterGMRES(i0,:)] = gmres(M, U, 100, 1e-4, 100);   % Generalized minimal residue method
% [F, flag(i0), relres(i0), iter(i0)] = tfqmr(M, U, [1e-6], length(U)*100);   % Transpose-free quasi-minimal residual method
% numErr(i0) = showGeneralCase(nodes, gNodes, F,...
%   nFullGrid, tempk, colorDataLim, PaperPosition, figHandle, stepQuiver,...
%   geoHandle, caseProperty, anlzHandle);
% numErr(i0) = checkError(nodes, gNodes, F,...
%   nFullGrid, tempk, caseProperty, anlzHandle);
% if exist('fig1', 'var')
%   close(fig1); clear fig1;
% end
% save(['.\', figHandle, '\', figHandle, '.mat']);

%--------------------------------------------------------------------------
% Helix locomotion case
% colorDataLim = [0, 1500];
% nFullGrid = [53, 53, 53];
% tempk = [3, 3, 1];
% stepQuiver = 4;
% w0 = 300*pi;
% v0 = linspace(0, 40, 11);
% PaperPosition = 1;
% shiftT = 0.6;
%
% nCase = length(v0);
% % if isempty(gcp('nocreate'))
% %   parpool(4);
% % end
% for i0 = 1:nCase
%   figHandle = sprintf('helixCase_Regularized_shiftT0.95_v%3.2e', v0(i0));
%   [nodes, gNodes, U, delta] = helixLocomotionCase(shiftT, w0, v0(i0));
% %   figHandle = sprintf('ellipseHelixCase_Regularized_shiftT0.95_v%3.2e', v0(i0));
% %   [nodesSphere, gNodesSphere, USphere] = ellipseMotionCase(shiftT, w0, v0(i0));
% %   [nodesHelix, gNodesHelix, UHelix, delta] = helixLocomotionCase(shiftT, w0, v0(i0));
% %   nodes = [nodesSphere; nodesHelix];
% %   gNodes = [gNodesSphere; gNodesHelix];
% %   U = [USphere; UHelix];
% %   fig1 = showProblem(nodes, 'gNodes', gNodes, 'U', U);
%
% %   stokesletProperty = struct('fNodes', gNodes);
% %   stokesletHandle = @stokesletMatrix3D;
%   stokesletProperty = struct('fNodes', nodes, 'epsilon', delta);
%   stokesletHandle = @RegularizedStokesletsMatrix3D;
%
%   M = stokesletHandle(nodes, stokesletProperty);
%   [F, flag(i0), relres(i0), iter(i0)] = tfqmr(M, U, [1e-6], length(U)*100);   % Transpose-free quasi-minimal residual method
%   showGeneralCase(nodes, stokesletHandle, stokesletProperty, F,...
%     nFullGrid, tempk, colorDataLim, PaperPosition, figHandle, stepQuiver,...
%     [], [], []);
%   if exist('fig1', 'var')
%     close(fig1);
%   end
% end

%--------------------------------------------------------------------------
% show numerical error
%
% shiftT = reshape(shiftT, n_nTracer, n_shiftT);
% nTracer = reshape(nTracer, n_nTracer, n_shiftT);
% numErr = reshape(numErr, n_nTracer, n_shiftT);
% flag = reshape(flag, n_nTracer, n_shiftT);
% relres = reshape(relres, n_nTracer, n_shiftT);
%
% figure;
% hold on;
% for i0 = 1:size(shiftT, 1)
%   DisplayName = ['N=', num2str(nTracer(i0, 1))];
%   semilogy(shiftT(i0, :), numErr(i0,:), 'DisplayName', DisplayName);
% %   DisplayName = ['N=', num2str(nTracer(i0, 1))];
% %   semilogy(shiftT(i0, :), relres(i0,:), 'DisplayName', DisplayName);
% end
% hold off
%
% log10nNode = log10(nTracer);
% log10numErr = log10(numErr);
% figure;
% hold on;
% conv = ones(size(nTracer, 2), 1);
% for i0 = 1:size(nTracer, 2)
%   DisplayName = ['?=', num2str(shiftT(1, i0))];
%   loglog(nTracer(:, i0), numErr(:, i0), 'DisplayName', DisplayName);
%   p = polyfit(log10nNode(:, i0), log10numErr(:, i0), 1);
%   conv(i0) = abs(p(1));
% end
% hold off
%
% figure;
% hold on;
% for i0 = 1:size(nTracer, 2)
%   DisplayName = ['relres=', num2str(shiftT(1, i0))];
%   loglog(nTracer(:, i0), relres(:, i0), 'DisplayName', DisplayName);
% end

%--------------------------------------------------------------------------
% Huang Mingji, 20160430
obj = struct('nodes', [],...
  'U', [],...
  'origin', [],...
  'node_index', [],...
  'freedom_index', []);
num_bacteria = 1;
w_body = 300*pi;
shiftT = 0.6;
bact_dist = 5;

colorDataLim = [];
num_plot_node = 1000;
tempk = [1, 3, 1];
stepQuiver = 4;
PaperPosition = 1;
nFullGrid = [round(sqrt(num_plot_node*(bact_dist*num_bacteria))),...
  round(sqrt(num_plot_node/(bact_dist*num_bacteria))), 53];

% geometric and boundary information
num_node = 0;
for i0 = 1:2:2*num_bacteria
  origin = [bact_dist*i0, 0, 0];
  % body
  [nodes, ~, ~] = ellipseMotionCase(shiftT, 0, 0);
  U = zeros(length(nodes(:)), 1);
  U(1:3:end) = -nodes(:,2) * w_body;
  U(2:3:end) = nodes(:,1) * w_body;
  nodes = bsxfun(@plus, nodes, origin);
  obj(i0).nodes = nodes;
  obj(i0).U = U;
  obj(i0).origin = origin;
  obj(i0).node_index = num_node+1 : num_node+length(nodes);
  obj(i0).freedom_index = 3*num_node+1 : 3*(num_node+length(nodes));
  num_node = num_node + length(nodes);
  % flagellum
  [nodes, ~, ~, delta] = helixLocomotionCase(shiftT, 0, 0);
  U = zeros(length(nodes(:)), 1);
  U(1:3:end) = -nodes(:,2) * w_body;
  U(2:3:end) = nodes(:,1) * w_body;
  nodes = bsxfun(@plus, nodes, origin);
  obj(i0+1).nodes = nodes;
  obj(i0+1).U = U;
  obj(i0+1).origin = origin;
  obj(i0+1).node_index = num_node+1 : num_node+length(nodes);
  obj(i0+1).freedom_index = 3*num_node+1 : 3*(num_node+length(nodes));
  num_node = num_node + length(nodes);
end
nodes = zeros(num_node, 3);
U = zeros(num_node*3, 1);
for i0 = 1:length(obj)
  nodes(obj(i0).node_index, :) = obj(i0).nodes;
  U(obj(i0).freedom_index) = obj(i0).U;
end
% fig1 = showProblem(nodes, 'U', U);

% construct M matrix
tic;
M = zeros( num_node*3 + num_bacteria*14 );
U = zeros( num_node*3 + num_bacteria*14, 1 );
% IMPORTANT: currently date structure only suited to one bacteria. Change
%  the structure of obj, store all informations of a bacteria using one
%  index, including the node and velocity indexs. 
temp1 = obj(2).freedom_index(end);    % final index of one bacteria. 
for i0 = 1:2
  stokesletProperty = struct('fNodes', obj(i0).nodes, 'epsilon', delta);
  M(obj(i0).freedom_index, obj(i0).freedom_index) = ...
    RegularizedStokesletsMatrix3D(obj(i0).nodes, stokesletProperty);
  for i1 = i0+1:2
    stokesletProperty = struct('fNodes', obj(i1).nodes, 'epsilon', delta);  
    M(obj(i0).freedom_index, obj(i1).freedom_index) = ...
      RegularizedStokesletsMatrix3D(obj(i0).nodes, stokesletProperty);
    M(obj(i1).freedom_index, obj(i0).freedom_index) = ...
      M(obj(i0).freedom_index, obj(i1).freedom_index)';
  end
  temp_nodes = obj(i0).nodes;
  temp_index = obj(i0).freedom_index;
  temp_num = length(obj(i0).freedom_index);
  M(temp_index(1:3:end), temp1+1) = 1;                     % I matrix
  M(temp_index(2:3:end), temp1+2) = 1;                     % I matrix
  M(temp_index(3:3:end), temp1+3) = -1;                    % I matrix
  M(temp_index(1:3:end), temp1+5) = temp_nodes(:, 3);      % L matrix
  M(temp_index(1:3:end), temp1+6) = -temp_nodes(:, 2);     % L matrix
  M(temp_index(2:3:end), temp1+6) = temp_nodes(:, 1);      % L matrix
  M(temp_index(2:3:end), temp1+4) = -temp_nodes(:, 3);     % L matrix
  M(temp_index(3:3:end), temp1+4) = temp_nodes(:, 2);      % L matrix
  M(temp_index(3:3:end), temp1+5) = -temp_nodes(:, 1);     % L matrix
  M(temp_index, temp1+7:temp1+9) = zeros(temp_num, 3);      % fill zero
  M(temp1+1:temp1+9, temp_index) = M(temp_index, temp1+1:temp1+9)'; 
  U(obj(i0).freedom_index) = obj(i0).U;
end
temp_index = obj(2).freedom_index;             % index of flagellum. 
M(temp_index(1:3:end), temp1+7) = -1;          % Lr matrix
M(temp_index(2:3:end), temp1+8) = -1;          % Lr matrix
M(temp1+1, temp1+9) = 1;           % zero conditions
M(temp1+2, temp1+10) = 1;          % zero conditions
M(temp1+4, temp1+11) = 1;          % zero conditions
M(temp1+5, temp1+12) = 1;          % zero conditions
M(temp1+6, temp1+13) = 1;          % zero conditions
M(temp1+9, temp1+14) = 1;          % zero conditions
M(temp1+9, temp1+1:temp1+9) = M(temp1+1:temp1+1, temp1+9);          % zero conditions
toc;
F = gmres(M, U, 100, 1e-12, 100);   % Generalized minimal residue method

r_index = 1;
for i0 = 1:length(obj)
  r_inc = length(obj(i0).U);
  F(r_index+r_inc:r_index+r_inc+2) = [];
  U(r_index+r_inc:r_index+r_inc+2) = [];
end
% fig2 = showProblem(nodes, 'U', F);
stokesletProperty = struct('fNodes', nodes, 'epsilon', delta);  
figHandle = 'HuangCase';
showGeneralCase(nodes, @RegularizedStokesletsMatrix3D, stokesletProperty, F,...
  nFullGrid, tempk, colorDataLim, PaperPosition, figHandle, stepQuiver,...
  [], [], []);

% nodes = [nodesSphere; nodesHelix];
% U = [USphere; UHelix];
% fig1 = showProblem(nodes, 'U', U);
% stokesletProperty = struct('fNodes', nodes, 'epsilon', delta);  
% 
% M = RegularizedStokesletsMatrix3D(nodes, stokesletProperty);
% 
% F = gmres(M, U, 100, 1e-6, 100);   % Generalized minimal residue method
% showGeneralCase(nodes, @RegularizedStokesletsMatrix3D, stokesletProperty, F,...
%   nFullGrid, tempk, colorDataLim, PaperPosition, figHandle, stepQuiver,...
%   [], [], []);
if exist('fig1', 'var')
  close(fig1);
end
if exist('fig2', 'var')
  close(fig2);
end

end









