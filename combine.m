function feat = combine(feat, comb_func, do_norm)
feat = comb_func(feat);
[along, across, num] = size(feat);
assert(along == 1);
feat = reshape(feat, [across, num])';
if do_norm
  feat = normalize(feat);
end

