% combine fun
% input: actually a 3-dim [combine_along_dim, combine_across_dim, num]
% output: a combined vector along dim 2 of size [combine_across_dim, num]
% -----------------------------------------------------------------------------
function feat = l2(diff)
% -----------------------------------------------------------------------------
feat = sqrt(sum(diff.*diff, 1));

