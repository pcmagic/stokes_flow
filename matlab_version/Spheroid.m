function spheroid_str = Spheroid(NSpheroid,shift,headA,headC)
% function to place points on one Ecoli surface
% the center of the Spheroid is at (0,0,0)
% Number of points NSpheroid
% shift is to place force points slightly away from bc points.
% major axis 2*headC, minor axis 2*headA
% Major axis along x
% rx ry rz are the locations of the boundary condition points
% gx gy gz are the locations of the force points
% date: 2015-10-22

% locating points on surface
jj = 1:NSpheroid;
xlocH = -1.0 + 2.0 * (jj - 1)/(NSpheroid - 1);
numf = 0.5;

mdl(1:NSpheroid) = 1.0;
prefac = 3.6*sqrt(headC/headA);
spherephi = ones(1, NSpheroid);
for ii = 1:NSpheroid
  if ii == 1 || ii == NSpheroid
    spherephi(ii) = 0;      % azimuthal angle
  else
    tr = sqrt(1-xlocH(ii)^2);
    wgt = prefac*(1.0-numf*(1.0-tr))/tr;
    spherephi(ii) = mod(spherephi(ii-1)+wgt/sqrt(NSpheroid),2.0*pi);
  end
end

tsin = sqrt(1-xlocH.^2);
rxS = headC*xlocH;
ryS = headA*tsin.*cos(spherephi);
rzS = headA*tsin.*sin(spherephi);

wgt2 = 1.0 - numf * (1.0 - tsin);
% wgt2 = 1.0-numf*tsin;
gxS = rxS.*(1-(1-shift)*wgt2.*mdl/headC);
gyS = ryS.*(1-(1-shift)*wgt2.*mdl/headA);
gzS = rzS.*(1-(1-shift)*wgt2.*mdl/headA);

spheroid_str = struct('rxS',rxS,'ryS',ryS,'rzS',rzS,'gxS',gxS,'gyS',gyS,'gzS',gzS);
end
