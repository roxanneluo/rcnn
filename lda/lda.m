function proj = lda(data, trans, proj_dim, filter_start)
dim = size(trans{1},1);
assert(dim ~= 0);
feat_dim = size(data, 2);
num_filter = feat_dim/dim;
assert(num_filter == length(trans));
proj = zeros(size(data,1), proj_dim);
parfor i = 1:num_filter
  fprintf('do lda %d\n', i);
  if ~isempty(trans{i})
    proj(:, filter_start(i):filter_start(i)+size(trans{i},2)-1) = ...
      data(:, (i-1)*dim+1:i*dim) * trans{i};   
  end
  fprintf('done lda %d\n', i);
end
