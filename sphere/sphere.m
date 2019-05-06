function sphere()

fileName = 'sphere_20161117.mat';
zoomMethod(fileName)
% exit

end

function independentMethod(fileName)
% Creat two node layers respectively. The outer one is velocity nodes and 
% the inter one is force nodes. The creation processes for them is 
% independent, without any contral. 

f_radias = 0.8;
u_radias = 1;
u_deltaLength = 0.07;
f_deltaLength = 0.14;
origin = [0, 0, 0];
u = [1, 0, 0, 0, 0, 0];     % [vx, vy, vz, wx, wy, wz]

fd=@(p) dsphere(p,origin(1), origin(2), origin(3), f_radias);
[f_nodes, f_mesh]=distmeshsurface(fd, @huniform, ...
                f_deltaLength, 1.1 * f_radias * [-1, -1, -1 ; 1, 1, 1]);

fd=@(p) dsphere(p,origin(1), origin(2), origin(3), u_radias);
[u_nodes, u_mesh] = distmeshsurface(fd, @huniform, ...
                u_deltaLength, 1.1 * u_radias * [-1, -1, -1 ; 1, 1, 1]);

U = u_nodes; 
U(:, 1) = u(1) + u(5) * (u_nodes(:, 3) - origin(3)) - u(6) * (u_nodes(:, 2) - origin(2));
U(:, 2) = u(2) + u(6) * (u_nodes(:, 1) - origin(1)) - u(4) * (u_nodes(:, 3) - origin(3));
U(:, 3) = u(3) + u(4) * (u_nodes(:, 2) - origin(2)) - u(5) * (u_nodes(:, 1) - origin(1));

save(fileName)

end

function zoomMethod(fileName)
% Create two layers with the same number of nodes. First creat velocity
% nodes, them create force nodes by zooming them in. 

u_radias = 1;
u_deltaLength = 0.1;
epsilon = 1;
origin = [0, 0, 0];
u = [1, 0, 0, 0, 0, 0];     % [vx, vy, vz, wx, wy, wz]

fd=@(p) dsphere(p,origin(1), origin(2), origin(3), u_radias);
[u_nodes, u_mesh] = distmeshsurface(fd, @huniform, ...
                u_deltaLength, 1.1 * u_radias * [-1, -1, -1 ; 1, 1, 1]);

factor = (u_radias - u_deltaLength * epsilon) / u_radias;
f_nodes = bsxfun(@plus, bsxfun(@minus, u_nodes, origin) * factor, origin);
f_mesh = u_mesh;

U = u_nodes; 
U(:, 1) = u(1) + u(5) * (u_nodes(:, 3) - origin(3)) - u(6) * (u_nodes(:, 2) - origin(2));
U(:, 2) = u(2) + u(6) * (u_nodes(:, 1) - origin(1)) - u(4) * (u_nodes(:, 3) - origin(3));
U(:, 3) = u(3) + u(4) * (u_nodes(:, 2) - origin(2)) - u(5) * (u_nodes(:, 1) - origin(1));

save(fileName)

end