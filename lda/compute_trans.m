function compute_trans(data, label, reduced_dim, filename)
fprintf('max %f, mean:%f, absmean: %f\n', ...
    max(max(data)), mean(data(:)), mean(abs(data(:))));
if exist(filename, 'file')
  fprintf('%s already exists\n', filename);
  return;
end

fprintf('LDAing\n');
data = double(data);
%disp(label)
[trans, val] = LDA(label, [], data, 19);
fprintf('dim of %s [%d]\n', filename, size(val,1));
save(filename, 'trans', 'val');
