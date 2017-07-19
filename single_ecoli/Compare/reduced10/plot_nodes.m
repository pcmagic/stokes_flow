% clear variables
% load('data.mat');

fhgeo = [bac1.gxT; bac1.gyT; bac1.gzT]';
vhgeo = [bac2.rxT; bac2.ryT; bac2.rzT]';
fsgeo = [bac1.gxH; bac1.gyH; bac1.gzH]';
vsgeo = [bac2.rxH; bac2.ryH; bac2.rzH]';
figure;
hold on;
plot3(vsgeo(:, 1), vsgeo(:, 2), vsgeo(:, 3),'.');
plot3(fsgeo(:, 1), fsgeo(:, 2), fsgeo(:, 3),'.');
plot3(vhgeo(:, 1), vhgeo(:, 2), vhgeo(:, 3),'.');
plot3(fhgeo(:, 1), fhgeo(:, 2), fhgeo(:, 3),'.');
hold off;
axis equal;