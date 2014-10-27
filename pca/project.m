
%-------------------------------------------------------
function proj = project(data, eigen_vec, eigen_val, whiten)
%-------------------------------------------------------
proj = data*eigen_vec;
if whiten
  fprintf('whitening\n');
  num_dim = size(data,2);
  inv = 1./sqrt(eigen_val+eps);
  proj = proj.*repmat(inv', [size(proj,1), 1]);
end
