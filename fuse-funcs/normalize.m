% -----------------------------------------------------------------------------
% input: a matrix [combine_across_dim, num]
% output: a matrix without 1
function feat = normalize(feat)
% -----------------------------------------------------------------------------
err = 1e-20;
s = sqrt(sum(feat.*feat,2));
s(abs(s) < err) = 1;
feat = feat./repmat(s,[1, size(feat,2)]);
