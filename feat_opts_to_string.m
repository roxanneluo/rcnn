function str = feat_opts_to_string(feat_opts)
str = feat_opt_to_string(feat_opts(1));
for i = 2:length(feat_opts)
  str = [str,'+',feat_opt_to_string(feat_opts(i))];
end

function str = feat_opt_to_string(feat_opt)
str = sprintf('l%d_%s_%s_%s', feat_opt.layer, feat_opt.b_or_d(1),...
    feat_opt.w_or_r(1), feat_opt.combine_name);

