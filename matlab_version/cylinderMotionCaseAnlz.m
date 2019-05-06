function [UxAnlz, UyAnlz] = cylinderMotionCaseAnlz(tempX, tempY, caseProperty)
% Give the analytical solution of flow past a cylinder.

a = caseProperty.radius;
center = caseProperty.center;
u0 = -caseProperty.u0;

tempX = tempX - center(1);
tempY = tempY - center(2);
r2 = tempX.^2 + tempY.^2;
r = sqrt(r2);
r(r<a) = NaN;
r2(r<a) = NaN;

f0x = 8 * pi/(1-2*log(a)) * u0;
f0y = 0;
temp1 = (2.*log(r)-a^2./r2) ./ (-8*pi);
temp2 = (1-a^2./r2) ./ (4*pi.*r2);
UxAnlz = temp1.*f0x + temp2.*(tempX.*tempX.*f0x + tempX.*tempY.*f0y);
UyAnlz = temp1.*f0y + temp2.*(tempX.*tempY.*f0x + tempY.*tempY.*f0y);

end
