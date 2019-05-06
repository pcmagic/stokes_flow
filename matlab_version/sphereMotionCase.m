function [nodes, gNodes, U, caseProperty] = sphereMotionCase(shiftT, nTracer)
% A sphere move with a constant velocity bodyUt1.
radius = 1.0;
alpha1 = 0.0;
u0 = 1;
locT1 = [0 0 0];

sphereObj = Spheroid(nTracer,shiftT,radius,radius);
oriT1 = [cos(alpha1) sin(alpha1) 0];
bodyUt1 = [u0 0.0 0.0 0.0 0.0 0.0];
sphereObj = BC_Spheroid(sphereObj,bodyUt1,locT1,oriT1);
nodes = [sphereObj.rxS; sphereObj.ryS; sphereObj.rzS]';
gNodes = [sphereObj.gxS; sphereObj.gyS; sphereObj.gzS]';    %ghost nodes
nNode = length(nodes(:));
U = ones(nNode, 1);                     % transpose of the velocity matrix
U(1:3:end) = sphereObj.uS;
U(2:3:end) = sphereObj.vS;
U(3:3:end) = sphereObj.wS;
center = mean(nodes);
caseProperty = struct('radius', radius,...
  'center', center,...
  'u0', u0);
end
