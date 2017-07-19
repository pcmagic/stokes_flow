%% ------------------------
% set boundary conditions on Ecoli surface
% bodyU(1:6) is the center of mass motion v(1:3) and omega(4:6)
% Ohead is the omega of head, relative to center of mass
% Otail is the omega of tail, relative to center of mass
% date: 2015-10-23

function rgv_str = BC_Ecoli(bac_str,bodyU,Ohead,Otail,locR)

% velocity on boundary
uhead = bodyU(5)*bac_str.rzH-bodyU(6)*bac_str.ryH+bodyU(1);   % total motion = body motion + relative motion
vhead = bodyU(6)*bac_str.rxH-(bodyU(4)+Ohead)*bac_str.rzH+bodyU(2);
whead = (bodyU(4)+Ohead)*bac_str.ryH-bodyU(5)*bac_str.rxH+bodyU(3);

utail = bodyU(5)*bac_str.rzT-bodyU(6)*bac_str.ryT+bodyU(1);   % total motion = body motion + relative motion
vtail = bodyU(6)*bac_str.rxT-(bodyU(4)+Otail)*bac_str.rzT+bodyU(2);
wtail = (bodyU(4)+Otail)*bac_str.ryT-bodyU(5)*bac_str.rxT+bodyU(3);

rxEcoli = [bac_str.rxH+locR(1) bac_str.rxT+locR(1)];
ryEcoli = [bac_str.ryH+locR(2) bac_str.ryT+locR(2)];
rzEcoli = [bac_str.rzH+locR(3) bac_str.rzT+locR(3)];

gxEcoli = [bac_str.gxH+locR(1) bac_str.gxT+locR(1)];
gyEcoli = [bac_str.gyH+locR(2) bac_str.gyT+locR(2)];
gzEcoli = [bac_str.gzH+locR(3) bac_str.gzT+locR(3)];

uEcoli = [uhead utail];
vEcoli = [vhead vtail];
wEcoli = [whead wtail];

rgv_str = struct('rxEcoli',rxEcoli,'ryEcoli',ryEcoli,'rzEcoli',rzEcoli,...
    'gxEcoli',gxEcoli,'gyEcoli',gyEcoli,'gzEcoli',gzEcoli,'uEcoli',uEcoli,'vEcoli',vEcoli,'wEcoli',wEcoli);