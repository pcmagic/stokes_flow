function [num_M, the_M, t_unodes, t_fnodes] = check_M()

num_M = [];
the_M = [];
% num_handle = 'eq02_016_0800_M_rank*';
% the_handle = 'eq02_num_016_0800_M_rank*';
% num_M = rebuild_var(num_handle, 'M');
% the_M = rebuild_var(num_handle, 'M');

obj_handle = 'eq02_016_0800_ecoli_';
fnodes = rebuild_var(obj_handle, 'fnodes');
fnodes = reshape(fnodes', [], 1);
f_glbIdx_all = rebuild_var(obj_handle, 'f_glbIdx_all') + 1;
t_fnodes = fnodes;
t_fnodes(f_glbIdx_all, 1) = reshape(repmat(fnodes(1:3:end), 1, 3)', [], 1);
t_fnodes(f_glbIdx_all, 2) = reshape(repmat(fnodes(2:3:end), 1, 3)', [], 1);
t_fnodes(f_glbIdx_all, 3) = reshape(repmat(fnodes(3:3:end), 1, 3)', [], 1);

unodes = rebuild_var(obj_handle, 'unodes');
unodes = reshape(unodes', [], 1);
u_glbIdx_all = rebuild_var(obj_handle, 'f_glbIdx_all') + 1;
t_unodes = unodes;
t_unodes(u_glbIdx_all, 1) = reshape(repmat(unodes(1:3:end), 1, 3)', [], 1);
t_unodes(u_glbIdx_all, 2) = reshape(repmat(unodes(2:3:end), 1, 3)', [], 1);
t_unodes(u_glbIdx_all, 3) = reshape(repmat(unodes(3:3:end), 1, 3)', [], 1);
end

function bigVar = rebuild_var(handle, varname)
if handle(end) ~= '*'
  handle = [handle, '*'];
end
Varnames = dir(handle);

n_mat = length(Varnames);
t_bigVar = cell(n_mat, 1);
for i0 = 1:n_mat
  t_name = Varnames(i0).name;
  load(t_name)
  eval(sprintf('t_bigVar{i0}=%s;', varname))
end

bigVar = cell2mat(t_bigVar);
end