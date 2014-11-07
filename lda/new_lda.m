function proj = lda(data, trans, proj_dim, filter_start)
dim = size(trans{1},1);
assert(dim ~= 0);
feat_dim = size(data, 2);
num_filter = feat_dim/dim;
assert(num_filter == length(trans));
proj_cell = cell(length(trans),1);
parfor i=1:length(trans)
  if ~isempty(trans{i})
    proj_cell{i} = data(:, (i-1)*dim+1:i*dim) * trans{i};   
  end
end
proj = zeros(size(data,1), proj_dim);
for i = 1:num_filter
  if ~isempty(trans{i})
    proj(:, filter_start(i):filter_start(i)+size(trans{i},2)-1) = ...
      proj_cell{i};
    proj_cell{i} = [];
  end
end
