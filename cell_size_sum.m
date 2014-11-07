function ssum = cell_size_sum(c, dim)
ssum = 0;
for i = 1:length(c)
  ssum = ssum + size(c{i}, dim);
end
