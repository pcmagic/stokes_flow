%% ------------------- 
% function to place points on one Ecoli surface
% the center of the Spheroid is at (0,0,0)
% Number of points NSpheroid
% shift is to place force points slightly away from bc points.
% major axis 2*headC, minor axis 2*headA
% Major axis along x
% rx ry rz are the locations of the boundary condition points
% gx gy gz are the locations of the force points
% date: 2015-10-22

function spheroid_str = Spheroid(NSpheroid,headA,headC,shift)

%% locating points on surface
jj = 1:NSpheroid;
 xlocH = -1.0 + 2.0*(jj-1)/(NSpheroid - 1.0);
 numf = 0.0;
 stepKK = 2.0/(NSpheroid-1);
%{
if headC/headA < 3.0
    xlocH = load('xloc_A2P5_mix_N1500.txt')';
    numf = 0.2;
    stepKK = 2.0/(NSpheroid-1);
else
    xlocH = load('xloc_A5_N2640_middle.txt')';
   % xlocH = load('xloc_A5_N2700.txt')';
    numf = 0.3;
   % stepKK = 2.0/(NSpheroid-1);
    stepKK = 2.0/(NSpheroid/2.2-1);
end
%}

mdl(1:NSpheroid) = 1.0;
%{
if headC/headA > 3.0
    for kk = 1:NSpheroid;
        if abs(xlocH(kk)) < 0.1
            mdl(kk) = sqrt(2.2 * 10);
        elseif abs(xlocH(kk)) < 0.2
            mdl(kk) = sqrt(2.2 * 4);
        else
            mdl(kk) = sqrt(2.2);
        end
    end
end
%}
prefac = 3.6*sqrt(headC/headA);

for ii = 1:NSpheroid
    if ii == 1 || ii == NSpheroid
        spherephi(ii) = 0;      % azimuthal angle
    else
        tdis = (xlocH(ii)-xlocH(ii-1))/stepKK;
        tr = sqrt(1-xlocH(ii)^2);
        spherephi(ii) = mod(spherephi(ii-1) + prefac/sqrt(NSpheroid)/tr, 2.0*pi);
      %  spherephi(ii) = mod(spherephi(ii-1) + tdis*prefac*(1.0-numf*tr)*mdl(ii)/sqrt(NSpheroid)/tr, 2.0*pi);
    end
end



tsin = sqrt(1-xlocH.^2);

rxS = (headC*xlocH);
ryS = (headA*tsin.*cos(spherephi));
rzS = (headA*tsin.*sin(spherephi));

gxS = rxS.*(1-(1-shift)*(1.0-numf*tsin).*mdl/headC);
gyS = ryS.*(1-(1-shift)*(1.0-numf*tsin).*mdl/headC);
gzS = rzS.*(1-(1-shift)*(1.0-numf*tsin).*mdl/headC);

spheroid_str = struct('rxS',rxS,'ryS',ryS,'rzS',rzS,'gxS',gxS,'gyS',gyS,'gzS',gzS);

%{
scrsz = get(0,'ScreenSize');
figure('Position',[1 1 scrsz(3)-1 scrsz(4)-1])
plot3(rxS,ryS,rzS,'-or',gxS,gyS,gzS,'-*b');
axis equal
axis([-11,15,-16,16,-3,3])
%}