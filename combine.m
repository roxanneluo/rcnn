function feat = combine(diff, comb_func, do_norm)
feat = comb_func(diff);
[along, across, num] = size(feat);
assert(along == 1);
feat = reshape(feat, [across, num]);
if do_norm
  feat = normalize(feat);
end
  

% -----------------------------------------------------------------------------
% input: a matrix [combine_across_dim, num]
% output: a matrix without 1
function feat = normalize(feat)
% -----------------------------------------------------------------------------
err = 1e-20;
s = sqrt(sum(feat.*feat,1));
s(abs(s) < err) = 1;
feat = feat./(ones(size(feat,1), 1)*s);
