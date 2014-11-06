function num_filter = get_num_filter(feat_opt)
assert(feat_opt.d && feat_opt.w);
assert(feat_opt.layer <= 5);
switch feat_opt.layer
case 5
  num_filter = 512;
case 4
  num_filter = 384;
otherwise
  assert(false);
end
