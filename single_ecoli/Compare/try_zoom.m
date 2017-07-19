tryzoom = load('try_zoom.txt');
prb_index = tryzoom(:, 1);
zoom_factor = tryzoom(:, 2);
non_dim_U = tryzoom(:, 3:8);
non_dim_F = tryzoom(:, 9:14);
velocity_err_sphere = tryzoom(:, 15:18);
velocity_err_helix0 = tryzoom(:, 19:22);
velocity_err_helix1 = tryzoom(:, 23:26);
velocity_err = tryzoom(:, 27:30);

figure;
loglog(zoom_factor, abs(non_dim_F), '*-');
t_legend = {
  'non_dim_F1',...
  'non_dim_F2',...
  'non_dim_F3',...
  'non_dim_M1',...
  'non_dim_M2',...
  'non_dim_M3'};
legend(t_legend, 'Interpreter', 'none');
xlabel('zoom\_factor');
ylabel({'F/0.5*(F_h+F_t)', ' ', 'M/(0.5*(M_h+M_t)* zoom\_factor)'});

figure;
semilogx(zoom_factor, non_dim_U, '*-');
t_legend = {
  'non_dim_U1',...
  'non_dim_U2',...
  'non_dim_U3',...
  'non_dim_W1',...
  'non_dim_W2',...
  'non_dim_W3'};
legend(t_legend, 'Interpreter', 'none');
xlabel('zoom\_factor');
ylabel({'U/zoom\_factor', ' ', 'W'});

figure;
loglog(zoom_factor, velocity_err_sphere, '*-')
t_legend = {
  'sphere_velocity_error_all',...
  'sphere_velocity_error_x',...
  'sphere_velocity_error_y',...
  'sphere_velocity_error_z'};
legend(t_legend, 'Interpreter', 'none');
xlabel('zoom\_factor');
ylabel('velocity\_error');
ylim([0.01, 1])

figure;
loglog(zoom_factor, [velocity_err_helix0, velocity_err_helix0], '*-')
t_legend = {
  'helix0_velocity_error_all',...
  'helix0_velocity_error_x',...
  'helix0_velocity_error_y',...
  'helix0_velocity_error_z',...
  'helix1_velocity_error_all',...
  'helix1_velocity_error_x',...
  'helix1_velocity_error_y',...
  'helix1_velocity_error_z'};
legend(t_legend, 'Interpreter', 'none');
xlabel('zoom\_factor');
ylabel('velocity\_error');
ylim([0.01, 1])

figure;
loglog(zoom_factor, velocity_err, '*-')
t_legend = {
  'velocity_error_all',...
  'velocity_error_x',...
  'velocity_error_y',...
  'velocity_error_z'};
legend(t_legend, 'Interpreter', 'none');
xlabel('zoom\_factor');
ylabel('velocity\_error');
ylim([0.01, 1])























