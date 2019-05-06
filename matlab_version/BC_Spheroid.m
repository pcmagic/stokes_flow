function rgv_str = BC_Spheroid(spheroid_str,bodyU,locT,oriT)
% set boundary conditions on Ecoli surface
% bodyU(1:6) is the center of mass motion v(1:3) and omega(4:6)
% locT(1:3) is the tracer center location
% oriT(1:3) is the orientation of tracer
% oriT(1) = cos(theta), oriT(2) = sin(theta)
% date: 2015-10-23

rxS = spheroid_str.rxS*oriT(1)-spheroid_str.ryS*oriT(2)+locT(1);
ryS = spheroid_str.rxS*oriT(2)+spheroid_str.ryS*oriT(1)+locT(2);
rzS = spheroid_str.rzS                                 +locT(3);

gxS = spheroid_str.gxS*oriT(1)-spheroid_str.gyS*oriT(2)+locT(1);
gyS = spheroid_str.gxS*oriT(2)+spheroid_str.gyS*oriT(1)+locT(2);
gzS = spheroid_str.gzS                                 +locT(3);

% velocity on boundary
uS = bodyU(5)*(rzS-locT(3))-bodyU(6)*(ryS-locT(2))+bodyU(1);
vS = bodyU(6)*(rxS-locT(1))-bodyU(4)*(rzS-locT(3))+bodyU(2);
wS = bodyU(4)*(ryS-locT(2))-bodyU(5)*(rxS-locT(1))+bodyU(3);

rgv_str = struct('rxS',rxS,'ryS',ryS,'rzS',rzS,...
  'gxS',gxS,'gyS',gyS,'gzS',gzS,'uS',uS,'vS',vS,'wS',wS);
end