function [nodes, gNodes, U] = cylinderRotationCase()
% A cylinder rotation with a constant angular velocity w0, anticlockwise is positive.
r = 1;
clength = 20;
w0 = 1*pi;
shiftT = 0.99999;
delta = 0.2;
% layer = [r*cos(0:delta:2*pi); r*sin(0:delta:2*pi)];
[nodesZ, theta] = meshgrid(0:delta:clength, 0:delta:2*pi);
nodes(1,:) = r*cos(theta(:));
nodes(2,:) = r*sin(theta(:));
nodes(3,:) = nodesZ(:);
% plot3(nodes(1,:), nodes(2,:), nodes(3,:)); axis equal;
gNodes(1,:) = r*shiftT*cos(theta(:));
gNodes(2,:) = r*shiftT*sin(theta(:));
gNodes(3,:) = nodesZ(:);
rotationMatrix = [cos(pi/2), -sin(pi/2), 0;
  sin(pi/2), cos(pi/2), 0;
  0,         0,         0];
U = w0 * rotationMatrix*nodes;

nodes = nodes'; gNodes = gNodes'; U = U(:);
end
