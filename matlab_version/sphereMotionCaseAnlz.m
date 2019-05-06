function [UxAnlz, UyAnlz, UzAnlz] = sphereMotionCaseAnlz(tempX, tempY, tempZ, caseProperty)
% Give the analytical solution of a sphere on the slice that Z coordinate
%  is a constant tempZ.

if min(tempZ(:)) ~= max(tempZ)
  errorMsg = 'tempZ must be a constant scalar or a vector or matrix with constant value. ';
  error('userError:sphereGeometry', errorMsg);
end
a = caseProperty.radius;
center = caseProperty.center;
u0 = -caseProperty.u0;

tempX = tempX - center(1);
tempY = tempY - center(2);
tempZ = tempZ - center(3);
nNode = length(tempX(:));

Rx2 = tempY.^2 + tempZ.^2;
Rx = sqrt(Rx2);
R2 = Rx2 + tempX.^2;
R = sqrt(R2);
sinTheta = Rx./R;
cosTheta = tempX./R;
sinPhi = tempZ./Rx;
cosPhi = tempY./Rx;
Ur =      u0.*cosTheta.*(1 - (3*a/2)./R + (a^3/2)./(R.*R2));
Utheta = -u0.*sinTheta.*(1 - (3*a/4)./R - (a^3/4)./(R.*R2));
Ur(R<a) = NaN;
Utheta(R<a) = NaN;
sinTheta(R<a) = NaN;
cosTheta(R<a) = NaN;
sinPhi(R<a) = NaN;
cosPhi(R<a) = NaN;

Utemp = ones(3, nNode);
for i0 = 1:nNode
  transMatrix = [sinTheta(i0)*cosPhi(i0), cosTheta(i0)*cosPhi(i0);
    sinTheta(i0)*sinPhi(i0), cosTheta(i0)*sinPhi(i0);
    cosTheta(i0), -sinTheta(i0)];
  
  Utemp(:, i0) = transMatrix * [Ur(i0); Utheta(i0)];
end
nNode2 = size(tempX);
UxAnlz = reshape(Utemp(3, :), nNode2) - u0;
UyAnlz = reshape(Utemp(1, :), nNode2);
UzAnlz = reshape(Utemp(2, :), nNode2);
end
