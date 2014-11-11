function filename = get_merged_trans_filename(feat_name, max_num_per_class, opts, suf)
if ~exist('suf', 'var')
  suf = 'mat';
end
filename = sprintf('lda/trans/%s/%d%s_TRANS.%s', feat_name, max_num_per_class, opts_name(opts), suf); 

