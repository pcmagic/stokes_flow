function [nodes, gNodes, U, epsilon] = helixLocomotionCase(shiftT, w0, u0)
% A helix rotation with a constant angular velocity w0, anticlockwise is
%  positive; and move with a constant velocity u0 along it's z axis.

r = [0.25, 0.01];
w = 1;
v = 0.3979;
perNodes = 8;

delta = 4*pi*r(2)/perNodes;
t = 0:3*delta/r(1):8*pi;
nodes = HelixDuplicate(r, w, v, t, 'perNodes', perNodes);
gNodes = HelixDuplicate(r.*[1,shiftT], w, v, t, 'perNodes', perNodes);
rotationMatrix = [w0*cos(pi/2), w0*sin(pi/2);
  -w0*sin(pi/2), w0*cos(pi/2)];
U = [rotationMatrix*nodes(1:2,:); u0*ones(size(nodes(3,:)))];

nodes = nodes'; gNodes = gNodes'; U = U(:);
epsilon = 2*r(2)/perNodes * 6/25;

end
