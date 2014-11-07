function trans = load_trans(feat_name, opts)
filename = get_merged_trans_filename(feat_name, opts.max_num_per_class, opts);
ld = load(filename);
trans = ld.trans; clear ld;


