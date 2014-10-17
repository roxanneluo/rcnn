function exist = exist_response(feat_opts)
exist = false;
for i = 1:length(feat_opts)
  exist = exist || (~feat_opts(i).w);
end
