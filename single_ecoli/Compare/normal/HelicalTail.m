%% ------------------- 
% function to place points on a helical tail
% tail starting from 0 to positive x
% rx ry rz are the locations of the boundary condition points
% gx gy gz are the locations of the force points
% Ntail = ppp*nP*circleN, number of points on the tail
% the0 is angular orientation of the first point of the helix
% date: 2015-10-22

function helix_str = HelicalTail(ppp,np,circleN,Ra,the0)
%% setting parameters

Ralpha = 0.2;   % radius of the helix
shift2 = 1.0;    % shift the points inside the boundary by a factor of "shift"
lamda = 3.0;    % helical pitch length


%% locating points on surface
wavek = 2*pi/lamda;
capD = sqrt(1+(Ralpha*wavek)^2);
capG = wavek;
dphi = 2*pi/circleN;
tp = ppp*np;

kk = 1:tp;
xlocT = lamda*kk/ppp;   % the x locations of points, starting from end of tail
theta = wavek*xlocT + the0;

centralx = xlocT;
centraly = Ralpha*cos(theta);
centralz = Ralpha*sin(theta);

for ii = 1:circleN;
    phi = ii*dphi;
    temp1 = Ralpha*Ra*capG*sin(phi)/capD;
    rxtail((ii-1)*tp+1:ii*tp) = xlocT + temp1;
    gxtail((ii-1)*tp+1:ii*tp) = xlocT + shift2*temp1;
    
    temp2 = (Ra/capG)*(wavek*sin(theta)*sin(phi)/capD + wavek*cos(theta)*cos(phi));
    rytail((ii-1)*tp+1:ii*tp) = Ralpha*cos(theta) + temp2;
    gytail((ii-1)*tp+1:ii*tp) = Ralpha*cos(theta) + shift2*temp2;
    
    temp3 = (Ra/capG)*(wavek*sin(theta)*cos(phi) - wavek*cos(theta)*sin(phi)/capD);
    rztail((ii-1)*tp+1:ii*tp) = Ralpha*sin(theta) + temp3;
    gztail((ii-1)*tp+1:ii*tp) = Ralpha*sin(theta) + shift2*temp3;
end

helix_str = struct('rxtail',rxtail,'rytail',rytail,'rztail',rztail,'gxtail',gxtail,'gytail',gytail,'gztail',gztail);

%{
scrsz = get(0,'ScreenSize');
figure('Position',[1 1 scrsz(3)-1 scrsz(4)-1])
plot3(centralx,centraly,centralz,'-or');
axis equal
axis([-11,15,-16,16,-3,3])
%}