function M = RegularizedStokesletsMatrix2D(nodes, fnodes, epsilon)

nNode = length(nodes);
nfNode = length(fnodes); 
M = zeros(nNode*2, nfNode*2);
for i0 = 1:nNode
  deltaX = fnodes(:, 1) - nodes(i0, 1);
  deltaY = fnodes(:, 2) - nodes(i0, 2);
  deltaR = sqrt(deltaX.^2 + deltaY.^2 + epsilon.^2);

  % see Eq.11 & Eq.14 (R. cortez, 2001) for details. 
  temp1 = deltaR + epsilon; 
  temp2 = (temp1+epsilon)./(temp1.*deltaR); 
  temp3 = -(log(temp1) - epsilon.*temp2)./(4*pi);
  temp4 = temp2./temp1./(4*pi);
  M(2*i0-1, 1:2:end) = temp3 + temp4.*deltaX.^2;
  M(2*i0-1, 2:2:end) =         temp4.*deltaX.*deltaY;
  M(2*i0,   1:2:end) =         M(2*i0-1, 2:2:end);
  M(2*i0,   2:2:end) = temp3 + temp4.*deltaY.^2;
end

end