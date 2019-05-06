function [x, y] = sphereMotionCaseGeometry(tempX, tempY, tempZ, caseProperty)
% Give the geometry of a sphere on the slice that Z coordinate is a constant tempZ.

if min(tempZ(:)) ~= max(tempZ)
  errorMsg = 'tempZ must be a constant scalar or a vector or matrix with constant value. ';
  error('userError:sphereGeometry', errorMsg);
end
radius = caseProperty.radius;
center = caseProperty.center;
x = []; y = [];

radius = sqrt(radius^2-(tempZ(1)-center(3))^2);
if ~isreal(radius)
  return;
end
theta = linspace(0, 2*pi, 64);
x = radius .* cos(theta);
y = radius .* sin(theta);
end
