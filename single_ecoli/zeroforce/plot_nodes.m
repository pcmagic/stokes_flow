clear variables
load('data.mat');

fhgeo = [bac1.gxT; bac1.gyT; bac1.gzT]';
vhgeo = [bac2.rxT; bac2.ryT; bac2.rzT]';
fsgeo = [bac1.gxH; bac1.gyH; bac1.gzH]';
vsgeo = [bac2.rxH; bac2.ryH; bac2.rzH]';
hold on;
plot3(vsgeo(1:100, 1), vsgeo(1:100, 2), vsgeo(1:100, 3),'.');
plot3(fsgeo(1:100, 1), fsgeo(1:100, 2), fsgeo(1:100, 3),'.');
hold off;
axis equal;