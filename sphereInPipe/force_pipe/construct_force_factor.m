clear variables

b_all = cell(0);
err_all = cell(0);
f1_list_all = cell(0);
f2_list_all = cell(0);
f3_list_all = cell(0);
residualNorm_all = cell(0);

filenames = dir('force_pipe0*mat');
for i0 = 1:length(filenames)
  filename = filenames(i0).name;
  load(filename);
  b_all{i0} = b;
  err_all{i0} = err;
  f1_list_all{i0} = f1_list;
  f2_list_all{i0} = f2_list;
  f3_list_all{i0} = f3_list;
  residualNorm_all{i0} = residualNorm;
end

b = cell2mat(b_all');
err = cell2mat(err_all');
f1_list = cell2mat(f1_list_all');
f2_list = cell2mat(f2_list_all');
f3_list = cell2mat(f3_list_all');
residualNorm = cell2mat(residualNorm_all');
save('construct09_force_pipe.mat', 'dp', 'lp', 'rp', 'ep', 'th', 'b', 'err',...
  'f1_list', 'f2_list', 'f3_list', 'residualNorm', 'stokesletsInPipe_pipeFactor');
