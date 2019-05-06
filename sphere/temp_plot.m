load('sphere_err_rs')
n = n + 1;
n_node = 400 * n .^ 2;
iter = n;
fig1 = figure;
axes1 = axes('Parent',fig1);
line0 = semilogy(1, 1,'Parent',axes1);
hold(axes1,'on');
for i0 = 1:n(end)
    iter(i0) = length(convergenceHistory{i0}) - 1;
    semilogy(0:iter(i0), convergenceHistory{i0},...
        'DisplayName', num2str(n_node(i0)))
end
delete(line0)
xlabel(axes1, 'number of iteration', 'interpreter', 'none');
ylabel(axes1, 'residual norm', 'interpreter', 'none');
box(axes1,'on');
set(axes1,'XGrid','on','XMinorGrid','on','XMinorTick','on',...
    'YGrid','on','YMinorGrid','on','YMinorTick','on');
legend(axes1,'show', 'Location', 'northeast');
hold off

fig1 = figure;
axes1 = axes('Parent',fig1);
plot(n_node, iter)
xlabel(axes1, 'number of nodes', 'interpreter', 'none');
ylabel(axes1, 'num of iteration', 'interpreter', 'none');
box(axes1,'on');
set(axes1,'XGrid','on','XMinorGrid','on','XMinorTick','on',...
    'YGrid','on','YMinorGrid','on','YMinorTick','on');
