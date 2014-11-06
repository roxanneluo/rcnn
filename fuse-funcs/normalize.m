% -----------------------------------------------------------------------------
% input: a matrix [combine_across_dim, num]
% output: a matrix without 1
function feat = normalize(feat)
% -----------------------------------------------------------------------------
s = sqrt(sum(feat.*feat,2));
s(abs(s) < eps) = 1;
feat = feat./repmat(s,[1, size(feat,2)]);
