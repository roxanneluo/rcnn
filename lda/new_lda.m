function proj = lda(data, trans, proj_dim, filter_start)
dim = size(trans{1},1);
assert(dim ~= 0);
feat_dim = size(data, 2);
num_filter = feat_dim/dim;
assert(num_filter == length(trans));
num = size(data,1);
proj = zeros(num, 19, length(trans));
em = false(length(trans));
tic
parfor i=1:length(trans)
  if ~isempty(trans{i})
    proj(:,:,i) = data(:, (i-1)*dim+1:i*dim)*trans{i};
  else 
    em(i) = true;
  end
end
toc
tic
proj(:,:,em) = [];
toc
tic
proj = reshape(proj, [num, numel(proj)/num]);
toc
