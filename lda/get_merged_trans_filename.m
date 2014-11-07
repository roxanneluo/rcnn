function filename = get_merged_trans_filename(feat_name, max_num_per_class, opts)
filename = sprintf('lda/trans/%s/%d%s_TRANS.mat', feat_name, max_num_per_class, opts_name(opts)); 

