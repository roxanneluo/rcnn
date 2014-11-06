function lda_feat = lda_trans(feat_opt, feat)
assert(feat_opt.w && feat_opt.d);
num_class = 20;
num_filter = get_num_filter(feat_opt);
lda_feat = zeros(size(feat,1), num_filter*num_class); 
dim = size(feat,2)/num_filter;
for i = 1:num_filter
  lda_feat(:, (i-1)*num_class+1:i*num_class) = ...
    feat(:,(i-1)*dim+1: i*dim) * lda.trans{i};
end
