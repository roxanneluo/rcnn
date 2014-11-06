function proj = lda(data, trans)
dim = size(trans{1},1);
assert(dim ~= 0);
feat_dim = size(data, 2);
num_filter = feat_dim/dim;
proj = [];
for i = 1:num_filter
  dim_start = (i-1)*dim+1;
  dim_end = i*dim;
  if ~isempty(trans{i})
    proj = cat(2, proj, data(:, dim_start:dim_end)*trans{i});
  end
end

