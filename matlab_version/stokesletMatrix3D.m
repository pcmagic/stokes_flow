function M = stokesletMatrix3D(nodes, stokesletMatrix3D)
% Solve M matrix
% Input:
%          nodes: velocity U node location infromation;
%          stokesletMatrix3D.fNodes: force F node location information.
% Output:
%          M: U = M * F.

gNodes = stokesletMatrix3D.fNodes;
nNode   = length(nodes);
ngNode = length(gNodes);
M = zeros(nNode*3, ngNode*3);
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