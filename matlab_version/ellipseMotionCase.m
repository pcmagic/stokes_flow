function [nodes, gNodes, U, caseProperty] = ellipseMotionCase(shiftT, w0, u0)
% A sphere rotation with a constant angular velocity w0, anticlockwise is
%  positive; and move with a constant velocity u0 along it's z axis.

nTracer = 300;
radius1 = 1;
radius2 = radius1*3;
alpha1 = 0.0;
locT1 = [0 0 0];

sphereObj = Spheroid(nTracer,shiftT,radius1,radius2);
oriT1 = [cos(alpha1) sin(alpha1) 0];
bodyUt1 = [0.0 0.0 u0 0.0 0.0 w0];
sphereObj = BC_Spheroid(sphereObj,bodyUt1,locT1,oriT1);
nodes = [-sphereObj.rzS; sphereObj.ryS; sphereObj.rxS-radius2;]';
gNodes = [-sphereObj.gzS; sphereObj.gyS; sphereObj.gxS-radius2;]';    %ghost nodes
nNode = length(nodes(:));
U = ones(nNode, 1);                     % transpose of the velocity matrix
U(1:3:end) = sphereObj.uS;
U(2:3:end) = sphereObj.vS;
U(3:3:end) = sphereObj.wS;
center = mean(nodes);
caseProperty = struct('radius1', radius1,...
  'radius2', radius2,...
  'center', center,...
  'u0', u0);
end
