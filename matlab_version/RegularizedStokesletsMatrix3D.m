function M = RegularizedStokesletsMatrix3D(nodes, stokesletMatrix3D)
% Solve M matrix using regularized stokeslets method
% Input:
%          nodes: velocity U node location infromation;
%          stokesletMatrix3D.fNodes:  force F node location information.
%          stokesletMatrix3D.epsilon: correction factor.
% Output:
%          M: U = M * F.

fNodes = stokesletMatrix3D.fNodes;
epsilon = stokesletMatrix3D.epsilon;
nNode   = size(nodes);
if nNode(1) < nNode(2)
    nodes = nodes';
end
nfNode   = size(fNodes);
if nfNode(1) < nfNode(2)
    fNodes = fNodes';
end
nfNode = length(fNodes);
nNode = length(nodes);

M = zeros(nNode*3, nfNode*3);
for i0 = 1:nNode
  delta_xi = bsxfun(@minus, fNodes, nodes(i0,:));
  temp1 = delta_xi.^2;
  delta_r2 = sum(temp1, 2) + epsilon^2;              % delta_r2 = r^2+e^2
  delta_r3 = delta_r2.^1.5;                    % delta_r3 = (r^2+e^2)^1.5
  temp2 = (delta_r2+epsilon^2) ./ delta_r3;    % temp2 = (r^2+2*e^2)/(r^2+e^2)^1.5
  M(3*i0-2, 1:3:end) = temp2 + delta_xi(:,1).*delta_xi(:,1)./delta_r3;   % Mxx
  M(3*i0-1, 2:3:end) = temp2 + delta_xi(:,2).*delta_xi(:,2)./delta_r3;   % Myy
  M(3*i0,   3:3:end) = temp2 + delta_xi(:,3).*delta_xi(:,3)./delta_r3;   % Mzz
  M(3*i0-2, 2:3:end) =         delta_xi(:,1).*delta_xi(:,2)./delta_r3;   % Mxy
  M(3*i0-2, 3:3:end) =         delta_xi(:,1).*delta_xi(:,3)./delta_r3;   % Mxz
  M(3*i0-1, 3:3:end) =         delta_xi(:,2).*delta_xi(:,3)./delta_r3;   % Myz
  M(3*i0-1, 1:3:end) = M(3*i0-2, 2:3:end);                          % Myx
  M(3*i0,   1:3:end) = M(3*i0-2, 3:3:end);                          % Mzx
  M(3*i0,   2:3:end) = M(3*i0-1, 3:3:end);                          % Mzy
end
end