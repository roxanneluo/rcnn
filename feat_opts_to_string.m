function str = feat_opts_to_string(feat_opts)
str = feat_opt_to_string(feat_opts(1));
for i = 2:length(feat_opts)
  str = [str,'+',feat_opt_to_string(feat_opts(i))];
end

function str = feat_opt_to_string(feat_opt)
if feat_opt.d
  bd = 'd';
else 
  bd = 'b';
end

if feat_opt.w
  wr = 'w';
else
  wr = 'r';
end

str = sprintf('l%d_%s_%s_%s', feat_opt.layer, bd, wr, feat_opt.combine_name);
if isfield(feat_opt, 'suf') && ~isempty(feat_opt.suf)
  str = [str '_' feat_opt.suf];
end

